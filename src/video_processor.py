import os
import cv2
import math
import time
import json
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from src.buffer import TemporalRingBuffer
from src.fsm import StateTransitionFSM
from src.agent import TemporalAgent

class VideoPipeline:
    def __init__(self):
        self.model = YOLO('yolov8n-pose.pt')
        self.obj_model = YOLO('yolov8n.pt')
        self.ring_buffer = TemporalRingBuffer(config_path="config/config.yaml")
        self.fsm = StateTransitionFSM(config_path="config/config.yaml")
        self.agent = TemporalAgent(log_path="outputs/agent_analysis.log")
        self.prev_x = None
        self.prev_y = None
        self.fps = self.fsm.config["tracker"]["fps"]
        self.state_frame_counts = defaultdict(int)

    def process_video(self, video_path, output_path="outputs/processed_video.mp4"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self.bed_bbox = None
        self.prev_x = None
        self.prev_y = None
        self.state_frame_counts = defaultdict(int)
        self.fsm = StateTransitionFSM(config_path="config/config.yaml")
        self.agent = TemporalAgent(log_path="outputs/agent_analysis.log")
        
        # Timeline generation tracking
        self.timeline = []
        last_timeline_state = None
        last_timeline_start_frame = 0

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or math.isnan(fps): fps = self.fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        mid_x, mid_y = width / 2, height / 2  

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            spine_angle = 0.0
            velocity = 0.0
            conf = 0.0
            in_bed = False
            aspect_ratio = 2.0
            current_state = "UNKNOWN"

            # ── 1. POSE DETECTION ──────────────────────────────────
            results = self.model.track(frame, persist=True, verbose=False)

            if results and results[0].keypoints is not None and len(results[0].keypoints) > 0:
                kpts = results[0].keypoints.data[0]   
                if len(kpts) >= 13:
                    s_l, s_r = kpts[5], kpts[6]
                    h_l, h_r = kpts[11], kpts[12]
                    conf = float((s_l[2] + s_r[2] + h_l[2] + h_r[2]) / 4.0)
                    
                    if conf > 0.3:
                        s_mid_x = float((s_l[0] + s_r[0]) / 2)
                        s_mid_y = float((s_l[1] + s_r[1]) / 2)
                        h_mid_x = float((h_l[0] + h_r[0]) / 2)
                        h_mid_y = float((h_l[1] + h_r[1]) / 2)
                        mid_x, mid_y = h_mid_x, h_mid_y

                        dx, dy = h_mid_x - s_mid_x, h_mid_y - s_mid_y
                        spine_angle = math.degrees(math.atan2(abs(dx), max(abs(dy), 1e-6)))

                        torso_len = max(math.sqrt(dx**2 + dy**2), 1.0)
                        if self.prev_x is not None:
                            dist = math.sqrt((mid_x - self.prev_x)**2 + (mid_y - self.prev_y)**2)
                            velocity = (dist / torso_len) * fps
                        self.prev_x, self.prev_y = mid_x, mid_y

                        if results[0].boxes and len(results[0].boxes) > 0:
                            bb = results[0].boxes[0].xywh[0]
                            bb_w, bb_h = max(float(bb[2]), 1.0), float(bb[3])
                            aspect_ratio = bb_h / bb_w
                            if float(bb[1]) + bb_h / 2.0 > height * 0.95: aspect_ratio = 2.0  

                        if self.bed_bbox is not None:
                            bx1, by1, bx2, by2 = self.bed_bbox
                            in_bed = (bx1 <= mid_x <= bx2) and (by1 <= mid_y <= by2)
                        else:
                            in_bed = mid_y > height * 0.6   

            # ── 2. DYNAMIC BED DETECTION ──
            if self.bed_bbox is None:
                obj_res = self.obj_model(frame, classes=[59], conf=0.1, verbose=False)
                if obj_res and obj_res[0].boxes and len(obj_res[0].boxes) > 0:
                    boxes = obj_res[0].boxes.xyxy.cpu().numpy()
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    self.bed_bbox = boxes[np.argmax(areas)]

            if self.bed_bbox is None and frame_idx > int(fps * 2):
                self.bed_bbox = [width * 0.1, height * 0.30, width * 0.9, height * 0.80]

            # ── 3. FSM STATE UPDATE ───────────────────────────────────────────
            current_state = self.fsm.update(spine_angle, velocity, conf, in_bed, aspect_ratio if conf > 0.3 else 2.0)
            self.state_frame_counts[current_state] += 1
            
            # Update Timeline Structure
            if current_state != last_timeline_state:
                if last_timeline_state is not None:
                    self.timeline.append({
                        "start_sec": last_timeline_start_frame / fps,
                        "end_sec": frame_idx / fps,
                        "state": last_timeline_state
                    })
                last_timeline_state = current_state
                last_timeline_start_frame = frame_idx

            # ── 4. AGENTIC EVENT DETECTION ────────────────────────────────────
            past = self.ring_buffer.get_past_segment(int(fps * 8))

            if current_state in ["WALKING", "OUT_OF_BED", "SITTING_OUTSIDE_BED"]:
                self.agent.evaluate_bed_exit_candidate(past, frame_idx, fps, current_state, conf)
            if current_state in ["LYING_IN_BED", "SITTING_ON_BED"]:
                self.agent.evaluate_return_to_bed(past, frame_idx, fps, current_state, conf)
            if current_state == "LYING_IN_BED" and frame_idx % int(fps * 3) == 0:
                self.agent.evaluate_lying_posture(mid_x, mid_y, self.bed_bbox, frame_idx)

            self.ring_buffer.append({"frame": frame_idx, "state": current_state, "spine": spine_angle, "in_bed": in_bed})

            # ── 5. DRAW ANNOTATIONS ───────────────────────────────────────────
            annotated_frame = results[0].plot() if results else frame
            if self.bed_bbox is not None:
                bx1, by1, bx2, by2 = [int(v) for v in self.bed_bbox]
                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (255, 80, 0), 2)

            label = f"State: {current_state}  Angle:{int(spine_angle)}  Vel:{velocity:.1f}"
            cv2.rectangle(annotated_frame, (0, 0), (width, 55), (0, 0, 0), -1)
            cv2.putText(annotated_frame, label, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 80), 2)
            out.write(annotated_frame)

        cap.release()
        out.release()
        
        # Close the last timeline segment
        if last_timeline_state is not None:
            self.timeline.append({"start_sec": last_timeline_start_frame / fps, "end_sec": total_frames / fps, "state": last_timeline_state})

        # ── 6. BUILD FINAL SUMMARIES ───────────────────────────────────────
        self._build_timeline_file()
        self._build_final_summary_json(total_frames / fps)
        self.agent.export_events()

        log_content = ""
        if os.path.exists("outputs/agent_analysis.log"):
            with open("outputs/agent_analysis.log", "r") as f: log_content = f.read()

        return output_path, log_content

    def _format_time(self, seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def _build_timeline_file(self):
        output_path = "outputs/timeline.txt"
        with open(output_path, "w") as f:
            for seg in self.timeline:
                start_str = self._format_time(seg["start_sec"])
                end_str = self._format_time(seg["end_sec"])
                f.write(f"{start_str} - {end_str}    {seg['state']}\n")

    def _build_final_summary_json(self, total_sec):
        output_path = "outputs/final_summary.json"
        
        activity_durations = {}
        for state, count in self.state_frame_counts.items():
            activity_durations[state.lower()] = int(count / self.fps)
            
        in_bed_sec = activity_durations.get("lying_in_bed", 0) + activity_durations.get("sitting_on_bed", 0)
        out_bed_sec = total_sec - in_bed_sec
        
        bed_exit_count = sum(1 for e in self.agent.events_list if e["event"] == "bed_exit")
        bed_return_count = sum(1 for e in self.agent.events_list if e["event"] == "bed_return")

        summary = {
            "observation_duration_sec": int(total_sec),
            "activity_duration_sec": activity_durations,
            "bed_exit_count": bed_exit_count,
            "bed_return_count": bed_return_count,
            "total_in_bed_sec": int(in_bed_sec),
            "total_out_of_bed_sec": int(out_bed_sec),
            "final_state": self.timeline[-1]["state"] if self.timeline else "UNKNOWN"
        }
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=4)
