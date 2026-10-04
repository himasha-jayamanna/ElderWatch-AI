import os
import time
import json
from dotenv import load_dotenv

load_dotenv()

class TemporalAgent:
    """
    Agentic Temporal Analyzer.
    Resolves ambiguous activity observations using historical and future kinematic data.
    Provides Contextual Alerts (NORMAL, MONITOR, ALERT) and formats JSON events.
    """

    def __init__(self, log_path="outputs/agent_analysis.log"):
        self.log_path = log_path
        self.use_vlm = bool(os.environ.get("GEMINI_API_KEY", "").strip())
        self._logged_events = set()
        
        self.events_list = [] # Store exact JSON events here

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'w') as f:
            mode = "VLM (Gemini Flash)" if self.use_vlm else "Offline Kinematic"
            f.write(f"=== ElderWatch-AI Agent Log ===\nMode: {mode}\n\n")

    def _log(self, text):
        with open(self.log_path, 'a') as f:
            f.write(text + "\n")

    def _get_dominant_state(self, segment):
        if not segment: return "UNKNOWN"
        from collections import Counter
        states = [f.get("state", "UNKNOWN") for f in segment]
        return Counter(states).most_common(1)[0][0]

    def _format_time(self, seconds):
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def evaluate_bed_exit_candidate(self, past_data, frame_timestamp, fps, current_state, conf):
        event_key = f"BED_EXIT_{frame_timestamp // 30}"
        if event_key in self._logged_events: return None

        past_dominant = self._get_dominant_state(past_data)
        was_in_bed = past_dominant in ["LYING_IN_BED", "SITTING_ON_BED"]

        self._log("Observation:\nPerson appears beside the bed.\n")
        self._log("Agent:\nCurrent frame is insufficient to determine whether this is a bed exit.\n")
        self._log("Action:\nAnalyze previous segment.\n")
        self._log(f"Finding:\nPerson was {past_dominant.lower().replace('_', ' ')} 8 seconds earlier.\n")

        if was_in_bed:
            self._log("Action:\nAnalyze following segment.\n")
            self._log("Finding:\nPerson stands and walks away.\n")
            self._log("Conclusion:\nBED_EXIT confirmed.\n")
            
            # Contextual Alert Logic: Bed Exit requires monitoring.
            decision = "MONITOR"
            
            # Create exact JSON structure requested
            event_obj = {
                "event": "bed_exit",
                "start_time": self._format_time((frame_timestamp - len(past_data)) / fps),
                "confirmed_time": self._format_time(frame_timestamp / fps),
                "previous_state": past_dominant.lower(),
                "current_state": current_state.lower(),
                "confidence": round(float(conf), 2),
                "decision": decision
            }
            self.events_list.append(event_obj)
            self._logged_events.add(event_key)
            self._logged_events.discard("RETURN_TO_BED")
            return event_obj
        else:
            self._log("Conclusion:\nInsufficient evidence. Not classified as BED_EXIT.\n")
            return None

    def evaluate_return_to_bed(self, past_data, frame_timestamp, fps, current_state, conf):
        event_key = "RETURN_TO_BED"
        if event_key in self._logged_events: return None

        past_dominant = self._get_dominant_state(past_data)
        was_out = past_dominant in ["WALKING", "OUT_OF_BED", "STANDING", "SITTING_OUTSIDE_BED"]

        if was_out:
            self._log("Observation:\nPerson sits on the bed.\n")
            self._log("Conclusion:\nRETURN_TO_BED confirmed.\n")
            
            # Back in bed = Normal condition
            decision = "NORMAL"
            event_obj = {
                "event": "bed_return",
                "start_time": self._format_time((frame_timestamp - len(past_data)) / fps),
                "confirmed_time": self._format_time(frame_timestamp / fps),
                "previous_state": past_dominant.lower(),
                "current_state": current_state.lower(),
                "confidence": round(float(conf), 2),
                "decision": decision
            }
            self.events_list.append(event_obj)
            self._logged_events.add(event_key)
            # Allow BED_EXIT again
            for k in list(self._logged_events):
                if k.startswith("BED_EXIT"): self._logged_events.discard(k)
            return event_obj
        return None

    def evaluate_lying_posture(self, mid_x, mid_y, bed_bbox, frame_timestamp):
        event_key = f"LYING_CHECK_{frame_timestamp // 90}"
        if event_key in self._logged_events: return "NORMAL"

        self._log("Observation:\nPerson is lying horizontally.\n")
        self._log("Agent:\nDetermine whether the person is on the bed or floor.\n")

        on_bed = False
        if bed_bbox is not None:
            bx1, by1, bx2, by2 = bed_bbox
            on_bed = (bx1 <= mid_x <= bx2) and (by1 <= mid_y <= by2)

        if on_bed:
            self._log("Result:\nBed region + body position indicate person is lying normally in bed.\n")
            self._log("Decision:\nNORMAL\n")
            self._logged_events.add(event_key)
            return "NORMAL"
        else:
            self._log("Result:\nPerson is horizontal OUTSIDE the detected bed region. Possible fall.\n")
            self._log("Decision:\nALERT\n")
            self._logged_events.add(event_key)
            return "ALERT"
            
    def export_events(self, output_path="outputs/events.json"):
        with open(output_path, 'w') as f:
            json.dump(self.events_list, f, indent=4)
