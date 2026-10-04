import gradio as gr
import os
from dotenv import load_dotenv
from src.video_processor import VideoPipeline

# Load .env so GEMINI_API_KEY is available system-wide before agent initialises
load_dotenv()

def process(video_file):
    if not video_file:
        return None, "⚠️  No video uploaded. Please upload an MP4 file."

    pipeline = VideoPipeline()
    out_vid, agent_log = pipeline.process_video(video_file)

    if not agent_log.strip():
        agent_log = "Processing complete. No significant events detected."

    return out_vid, agent_log

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="ElderWatch-AI",
    theme=gr.themes.Base(primary_hue="blue"),
    css="""
        .gr-button-primary { font-size: 1.1rem !important; padding: 12px !important; }
        footer { display: none !important; }
    """
) as demo:

    gr.Markdown("# 🏥 ElderWatch-AI — Elderly Patient Monitoring System")
    gr.Markdown(
        "Upload a short indoor video. The system uses **YOLOv8n-Pose**, "
        "**ByteTrack**, a **Finite State Machine**, and an **Agentic Temporal Reasoner** "
        "to track activity states, detect bed exits / returns, and log reasoning."
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_in    = gr.Video(label="📂 Input Video (MP4)")
            analyze_btn = gr.Button("▶  Analyse Video", variant="primary")

        with gr.Column(scale=1):
            video_out = gr.Video(label="📽  Analysed Output — Skeletons + FSM State Overlay")

    with gr.Row():
        log_out = gr.Textbox(
            label="🤖 Agent Reasoning Log  (outputs/agent_analysis.log)",
            lines=22,
            max_lines=50
        )

    analyze_btn.click(fn=process, inputs=[video_in], outputs=[video_out, log_out])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
