from collections import deque
import yaml

class TemporalRingBuffer:
    """
    A sliding window memory structure (Ring Buffer).
    It retains temporal context (past and future segments) so the 
    Agentic System can analyze confusing states (e.g., blanket occlusions).
    """

    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.fps = self.config["tracker"]["fps"]
        # How far back and forward to remember for the Agent
        past_sec = self.config["buffer"]["past_history_seconds"]
        fut_sec = self.config["buffer"]["future_lookahead_seconds"]
        
        # Total ring buffer frame capacity (past + future capacity)
        self.max_size = int(self.fps * (past_sec + fut_sec + 1))
        
        # deque automatically discards oldest data when max_size is reached
        self.buffer = deque(maxlen=self.max_size)

    def append(self, frame_data):
        """
        Pushes a new frame's data (bounding box, pose angles, etc.) into the buffer.
        """
        self.buffer.append(frame_data)

    def get_past_segment(self, frames_back):
        """
        Fetches the historical segment from t - frames_back to t (current).
        """
        data_list = list(self.buffer)
        if len(data_list) == 0:
            return []
        # Return the last 'frames_back' elements up to the current
        return data_list[-frames_back:]

    def get_full_buffer(self):
        """ Returns the entire stored memory tape for deep analysis """
        return list(self.buffer)
