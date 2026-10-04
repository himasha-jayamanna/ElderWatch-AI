"""
ElderWatch-AI — Evaluation Script
==================================
Compares system FSM predictions against manually defined ground truth annotations.

Usage:
    python tests/evaluate.py --video test_videos/video_01_basic_states.mp4

Output:
    - Per-state Precision, Recall, F1-Score
    - Overall Accuracy
    - Event Detection Rate (BED_EXIT, RETURN_TO_BED)
    - Saves results to outputs/evaluation_report.txt
"""
import argparse
import json
import os
import sys
from collections import defaultdict

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.video_processor import VideoPipeline


GROUND_TRUTH_PATH = "tests/ground_truth.json"
REPORT_PATH       = "outputs/evaluation_report.txt"


def build_frame_labels(segments, fps, total_frames):
    """
    Converts time-range segments into a per-frame label list.
    Example: [{start_sec:0, end_sec:3, state:'LYING_IN_BED'}] → ['LYING_IN_BED', 'LYING_IN_BED', ...]
    """
    labels = ["UNKNOWN"] * total_frames
    for seg in segments:
        start_f = int(seg["start_sec"] * fps)
        end_f   = min(int(seg["end_sec"] * fps), total_frames)
        for i in range(start_f, end_f):
            labels[i] = seg["expected_state"]
    return labels


def compute_metrics(gt_labels, pred_labels, all_states):
    """
    Computes per-state Precision, Recall, F1, and overall Accuracy.

    Precision  = True Positives / (True Positives + False Positives)
               = "When system said X, how often was it right?"
    Recall     = True Positives / (True Positives + False Negatives)
               = "Of all actual X moments, how many did the system catch?"
    F1 Score   = Harmonic mean of Precision and Recall (balanced measure)
    """
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    correct = 0
    for gt, pred in zip(gt_labels, pred_labels):
        if gt == pred:
            correct += 1
            tp[gt]  += 1
        else:
            fp[pred] += 1
            fn[gt]   += 1

    accuracy = correct / max(len(gt_labels), 1)

    per_state = {}
    for state in all_states:
        p  = tp[state] / max(tp[state] + fp[state], 1)
        r  = tp[state] / max(tp[state] + fn[state], 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        per_state[state] = {"precision": p, "recall": r, "f1": f1,
                            "tp": tp[state], "fp": fp[state], "fn": fn[state]}
    return accuracy, per_state


def run_pipeline_and_collect(video_path):
    """
    Runs the full ElderWatch pipeline on the video and collects per-frame state predictions.
    Returns (predictions list, set of agent-logged events, fps).
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Process video — the pipeline writes the output video + agent log
    pipeline = VideoPipeline()
    pipeline.process_video(video_path)

    # Read state sequence from the ring buffer history stored during processing
    pred_labels = [d["state"] for d in pipeline.ring_buffer.get_full_buffer()]

    # Pad or trim to match total_frames
    if len(pred_labels) < total_frames:
        pred_labels += ["UNKNOWN"] * (total_frames - len(pred_labels))
    pred_labels = pred_labels[:total_frames]

    # Parse logged events from agent log
    logged_events = set()
    log_path = "outputs/agent_analysis.log"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            content = f.read()
        if "BED_EXIT confirmed"      in content: logged_events.add("BED_EXIT")
        if "RETURN_TO_BED confirmed" in content: logged_events.add("RETURN_TO_BED")

    return pred_labels, logged_events, fps, total_frames


def evaluate(video_path):
    video_name = os.path.basename(video_path)

    with open(GROUND_TRUTH_PATH, "r") as f:
        gt_data = json.load(f)

    if video_name not in gt_data:
        print(f"[ERROR] No ground truth entry found for '{video_name}' in {GROUND_TRUTH_PATH}")
        print("Please add an entry to tests/ground_truth.json first.")
        return

    gt_entry = gt_data[video_name]

    print(f"\n{'='*55}")
    print(f" ElderWatch-AI Evaluation — {video_name}")
    print(f"{'='*55}")
    print("Running pipeline... (this may take a moment)")

    pred_labels, logged_events, fps, total_frames = run_pipeline_and_collect(video_path)

    # Override fps with ground truth fps if provided
    fps = gt_entry.get("fps", fps)
    gt_labels = build_frame_labels(gt_entry["segments"], fps, total_frames)

    all_states = list({s for s in gt_labels + pred_labels if s != "UNKNOWN"})
    accuracy, per_state = compute_metrics(gt_labels, pred_labels, all_states)

    expected_events = set(gt_entry.get("expected_events", []))
    event_hits      = expected_events & logged_events
    event_misses    = expected_events - logged_events
    event_rate      = len(event_hits) / max(len(expected_events), 1)

    # ── Print report ────────────────────────────────────────────────────────
    lines = []
    lines.append(f"\nOverall Frame Accuracy : {accuracy*100:.1f}%")
    lines.append(f"\n{'State':<25} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    lines.append("-" * 55)
    for state, m in sorted(per_state.items()):
        lines.append(f"{state:<25} {m['precision']*100:>9.1f}%  {m['recall']*100:>6.1f}%  {m['f1']*100:>6.1f}%")

    lines.append(f"\nEvent Detection Rate   : {event_rate*100:.1f}%")
    lines.append(f"  Events Detected      : {sorted(event_hits)}")
    lines.append(f"  Events Missed        : {sorted(event_misses)}")
    lines.append(f"  Unexpected Events    : {sorted(logged_events - expected_events)}")
    lines.append(f"\n{'='*55}\n")

    report = "\n".join(lines)
    print(report)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(f"ElderWatch-AI Evaluation Report\nVideo: {video_name}\n")
        f.write(report)
    print(f"Report saved → {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ElderWatch-AI Evaluation Script")
    parser.add_argument("--video", required=True, help="Path to the test video file")
    args = parser.parse_args()
    evaluate(args.video)
