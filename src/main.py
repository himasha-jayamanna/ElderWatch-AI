from buffer import TemporalRingBuffer
from fsm import StateTransitionFSM
from agent import TemporalAgent
import time

def main():
    """
    Main orchestration loop for ElderWatch-AI.
    Simulates video ingestion, YOLOv8n tracking, FSM state mapping, and Agentic analysis.
    """
    print("Initializing ElderWatch-AI Pipeline...")
    
    # Initialize Core Modules
    ring_buffer = TemporalRingBuffer(config_path="config/config.yaml")
    fsm = StateTransitionFSM(config_path="config/config.yaml")
    agent = TemporalAgent(log_path="outputs/agent_analysis.log")
    
    # Simulated frame loop replacing a heavy video file for structural testing
    print("Starting Video Analysis Simulator...")
    
    # Simulate a scenario: Lying in bed -> Standing beside bed -> Walking away
    simulation_data = [
        {"frame": 1, "spine_angle": 80.0, "vel": 0.0, "conf": 0.9, "in_bed": True},    # Lying
        {"frame": 30, "spine_angle": 75.0, "vel": 0.1, "conf": 0.9, "in_bed": True},   # Lying
        {"frame": 60, "spine_angle": 45.0, "vel": 0.5, "conf": 0.8, "in_bed": True},   # Sitting
        {"frame": 90, "spine_angle": 20.0, "vel": 0.2, "conf": 0.9, "in_bed": False},  # Standing beside bed
        {"frame": 150,"spine_angle": 15.0, "vel": 2.0, "conf": 0.9, "in_bed": False},  # Walking away
    ]

    for data in simulation_data:
        # 1. Update the Memory Buffer
        ring_buffer.append(data)
        
        # 2. Update the FSM
        current_state = fsm.update(
            spine_angle=data["spine_angle"],
            velocity=data["vel"],
            pose_conf=data["conf"],
            in_bed_zone=data["in_bed"]
        )
        
        # 3. Agent checks for triggers
        # If FSM transitions to standing off bed, agent needs to verify if it's a bed exit
        if data["frame"] == 90:
            past = ring_buffer.get_past_segment(frames_back=90)
            agent.evaluate_bed_exit_candidate(past, following_data=[])
            
        elif data["frame"] == 1:
            agent.evaluate_lying_posture()

        time.sleep(0.1) # Simulate real-time processing gap

    print("Analysis Complete. Check outputs/agent_analysis.log for Agent reasoning.")

if __name__ == "__main__":
    main()
