import yaml
from collections import deque, Counter

class StateTransitionFSM:
    """
    Finite State Machine for robust health activity recognition.
    Uses majority voting over a sliding time window to prevent frame-by-frame flickering.
    Think of it like a jury: a state only becomes 'official' when >60% of recent frames agree.
    """
    
    STATES = [
        "LYING_IN_BED", "SITTING_ON_BED", "SITTING_OUTSIDE_BED", 
        "STANDING", "WALKING", "OUT_OF_BED", "UNKNOWN"
    ]

    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        fps = self.config["tracker"]["fps"]
        window_sec = self.config["fsm"]["majority_vote_window_seconds"]
        self.window_frames = int(fps * window_sec)
        
        # Rolling buffer of raw per-frame state predictions
        self.prediction_history = deque(maxlen=self.window_frames)
        
        # The officially confirmed state (only changes after majority vote passes)
        self.current_state = "UNKNOWN"
        
        # Track the previous confirmed state for transition analysis
        self.previous_state = "UNKNOWN"

    def estimate_raw_state(self, spine_angle, velocity, pose_conf, in_bed_zone, aspect_ratio=2.0):
        """
        Calculates an instantaneous activity estimate from geometric features.
        
        Spine Angle: angle in degrees between spine vector and vertical axis.
          - >60°  → roughly horizontal → person is Lying Down
          - 30-60° → semi-upright        → person is Sitting
          - <30°  → mostly upright       → Standing or Walking
        
        Aspect Ratio (Bounding Box H/W):
          - If spine is upright (<30°) but box is wide/squarish (<1.5 ratio), person is probably sitting upright.
          - Tall box (≥1.5) with upright spine means truly Standing or Walking.
        
        Velocity (normalised to torso-heights per second):
          - >0.5 → person centre-of-mass is moving fast enough to be Walking
          - ≤0.5 → person is relatively still → Standing
        """
        conf_thresh = self.config["agent"]["min_pose_confidence"]
        
        if pose_conf < conf_thresh:
            return "UNKNOWN"

        lying_thresh  = self.config["fsm"]["spine_angle_lying_thresh"]    # 60°
        standing_thresh = self.config["fsm"]["spine_angle_standing_thresh"]  # 30°
        move_thresh   = self.config["fsm"]["movement_velocity_thresh"]     # 0.5

        if spine_angle >= lying_thresh:
            # Horizontal posture — in bed or on floor (agent resolves which)
            return "LYING_IN_BED" if in_bed_zone else "UNKNOWN"

        elif standing_thresh < spine_angle < lying_thresh:
            # Semi-upright torso → Sitting
            return "SITTING_ON_BED" if in_bed_zone else "SITTING_OUTSIDE_BED"

        elif spine_angle <= standing_thresh:
            # Upright torso. Decide Sit vs Stand via bounding-box aspect ratio.
            # A squarish box (H/W < 1.5) usually means the person is seated upright.
            if aspect_ratio < 1.5:
                return "SITTING_ON_BED" if in_bed_zone else "SITTING_OUTSIDE_BED"
            else:
                # Tall box → person is fully standing. Movement decides Walk vs Stand.
                if velocity > move_thresh:
                    return "WALKING"
                else:
                    return "OUT_OF_BED" if not in_bed_zone else "STANDING"

        return "UNKNOWN"

    def update(self, spine_angle, velocity, pose_conf, in_bed_zone, aspect_ratio=2.0):
        """
        Runs the majority-vote smoother and updates the confirmed state.
        Returns the current confirmed state.
        """
        raw_state = self.estimate_raw_state(spine_angle, velocity, pose_conf, in_bed_zone, aspect_ratio)
        self.prediction_history.append(raw_state)

        # Only commit a state transition when the window is full
        if len(self.prediction_history) == self.window_frames:
            counts = Counter(self.prediction_history)
            dominant_state, dominant_count = counts.most_common(1)[0]
            
            # Require >60% agreement in the window before changing state
            if dominant_count > (self.window_frames * 0.6):
                if dominant_state != self.current_state:
                    self.previous_state = self.current_state
                self.current_state = dominant_state

        return self.current_state
