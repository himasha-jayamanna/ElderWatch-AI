"""
ElderWatch-AI — Full System Batch Test Runner
==============================================
Runs all test videos through the complete pipeline and generates a consolidated
evaluation report comparing system predictions against ground truth annotations.

Usage:
    python run_tests.py

Output:
    - outputs/processed_video_<name>.mp4  (annotated video per input)
    - outputs/agent_analysis_<name>.log   (agent reasoning per video)
    - evaluation_results/FULL_EVALUATION_REPORT.txt  (consolidated metrics for all videos)
"""
import os
import sys
import json
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.video_processor import VideoPipeline

GROUND_TRUTH_PATH  = "tests/ground_truth.json"
TEST_VIDEO_DIR     = "test_videos"
REPORT_PATH        = "evaluation_results/FULL_EVALUATION_REPORT.txt"


def build_frame_labels(segments, fps, total_frames):
    labels = ["UNKNOWN"] * total_frames
    for seg in segments:
        start_f = int(seg["start_sec"] * fps)
        end_f   = min(int(seg["end_sec"] * fps), total_frames)
        for i in range(start_f, end_f):
            labels[i] = seg["expected_state"]
    return labels


def compute_metrics(gt_labels, pred_labels, all_states):
    """
    Precision  = When system said X, how often it was correct.
    Recall     = Of all actual X moments, how many system caught.
    F1         = Balanced score between Precision and Recall.
    Accuracy   = Overall correct predictions / total frames.
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
    duration_errors = {}
    for state in all_states:
        p  = tp[state] / max(tp[state] + fp[state], 1)
        r  = tp[state] / max(tp[state] + fn[state], 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        per_state[state] = {"precision": p, "recall": r, "f1": f1}
        
        gt_frames = tp[state] + fn[state]
        pred_frames = tp[state] + fp[state]
        duration_errors[state] = abs(pred_frames - gt_frames)

    return accuracy, per_state, duration_errors


def run_single_video(video_path, gt_entry, pipeline):
    import cv2
    video_name = os.path.basename(video_path)
    stem       = os.path.splitext(video_name)[0]

    out_video   = f"outputs/processed_{stem}.mp4"
    out_log     = f"outputs/agent_analysis_{stem}.log"

    # Redirect agent log to per-video file
    pipeline.agent.log_path = out_log
    with open(out_log, 'w') as f:
        mode = "VLM (Gemini Flash)" if pipeline.agent.use_vlm else "Offline Kinematic"
        f.write(f"=== ElderWatch-AI Agent Log — {video_name} ===\nMode: {mode}\n\n")
    pipeline.agent._logged_events.clear()

    print(f"  Processing: {video_name} ...")
    _, _ = pipeline.process_video(video_path, output_path=out_video)

    # Collect predictions from ring buffer
    pred_labels = [d["state"] for d in pipeline.ring_buffer.get_full_buffer()]

    # Get total frames from video
    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or gt_entry.get("fps", 30)
    cap.release()

    fps = gt_entry.get("fps", fps)
    if len(pred_labels) < total_frames:
        pred_labels += ["UNKNOWN"] * (total_frames - len(pred_labels))
    pred_labels = pred_labels[:total_frames]

    gt_labels = build_frame_labels(gt_entry["segments"], fps, total_frames)

    # Parse events from log  
    logged_events = set()
    if os.path.exists(out_log):
        with open(out_log, "r") as f:
            content = f.read()
        if "BED_EXIT confirmed"      in content: logged_events.add("BED_EXIT")
        if "RETURN_TO_BED confirmed" in content: logged_events.add("RETURN_TO_BED")

    expected_events = set(gt_entry.get("expected_events", []))
    all_states      = list({s for s in gt_labels + pred_labels if s != "UNKNOWN"})
    accuracy, per_state, duration_errors_frames = compute_metrics(gt_labels, pred_labels, all_states)

    event_hits  = expected_events & logged_events
    event_miss  = expected_events - logged_events
    event_rate  = len(event_hits) / max(len(expected_events), 1) if expected_events else 1.0

    duration_error_text = []
    for state, err_frames in duration_errors_frames.items():
        err_sec = int(err_frames / fps)
        state_name = state.lower().replace("_", " ").capitalize()
        duration_error_text.append(f" {state_name} duration error: {err_sec} sec")

    return {
        "video"          : video_name,
        "accuracy"       : accuracy,
        "per_state"      : per_state,
        "duration_errors": duration_error_text,
        "event_rate"     : event_rate,
        "events_detected": sorted(event_hits),
        "events_missed"  : sorted(event_miss),
        "events_extra"   : sorted(logged_events - expected_events),
        "out_video"      : out_video,
        "out_log"        : out_log,
    }


def print_result(r):
    sep = "=" * 60
    lines = [
        sep,
        f" Video         : {r['video']}",
        f" Frame Accuracy: {r['accuracy']*100:.1f}%",
        f" Event Rate    : {r['event_rate']*100:.1f}%",
        f" Events Found  : {r['events_detected']}",
        f" Events Missed : {r['events_missed']}",
        f" Extra Events  : {r['events_extra']}",
        "",
        f" {'State':<28} {'Precision':>9} {'Recall':>8} {'F1':>7}",
        "-" * 60,
    ]
    for state, m in sorted(r["per_state"].items()):
        lines.append(
            f" {state:<28} {m['precision']*100:>8.1f}%  {m['recall']*100:>6.1f}%  {m['f1']*100:>6.1f}%"
        )
    lines.append("\n Duration Estimation Errors:")
    lines.extend(r["duration_errors"])
    lines += ["", f" Output Video  : {r['out_video']}", f" Agent Log     : {r['out_log']}", sep]
    return "\n".join(lines)


def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("evaluation_results", exist_ok=True)
    os.makedirs(TEST_VIDEO_DIR, exist_ok=True)

    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"[ERROR] Ground truth file not found: {GROUND_TRUTH_PATH}")
        return

    with open(GROUND_TRUTH_PATH, "r") as f:
        gt_data = json.load(f)

    # Collect test videos
    video_files = sorted([
        os.path.join(TEST_VIDEO_DIR, f)
        for f in os.listdir(TEST_VIDEO_DIR)
        if f.lower().endswith(".mp4")
    ])

    if not video_files:
        print(f"[ERROR] No MP4 files found in '{TEST_VIDEO_DIR}/'.")
        print("Please add your test videos there and re-run.")
        return

    print(f"\n{'='*60}")
    print(f"  ElderWatch-AI — Full Batch Evaluation")
    print(f"  Videos found : {len(video_files)}")
    print(f"  Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # One shared pipeline instance (models loaded once = faster)
    pipeline = VideoPipeline()
    all_results = []

    for video_path in video_files:
        vname = os.path.basename(video_path)
        if vname not in gt_data:
            print(f"  [SKIP] No ground truth entry for '{vname}'. Add it to tests/ground_truth.json.")
            continue

        result = run_single_video(video_path, gt_data[vname], pipeline)
        all_results.append(result)
        print(print_result(result))

    if not all_results:
        print("No videos were evaluated. Check ground truth entries match your filenames.")
        return

    # ── Consolidated Summary ─────────────────────────────────────────────────
    avg_acc   = sum(r["accuracy"]   for r in all_results) / len(all_results)
    avg_event = sum(r["event_rate"] for r in all_results) / len(all_results)

    summary = [
        "\n" + "="*60,
        f"  CONSOLIDATED SUMMARY  ({len(all_results)} videos)",
        "="*60,
        f"  Average Frame Accuracy : {avg_acc*100:.1f}%",
        f"  Average Event Rate     : {avg_event*100:.1f}%",
        "="*60 + "\n",
    ]
    summary_str = "\n".join(summary)
    print(summary_str)

    # Write full report
    full_report = "\n".join([
        f"ElderWatch-AI Full Evaluation Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ] + [print_result(r) for r in all_results] + [summary_str])

    with open(REPORT_PATH, "w") as f:
        f.write(full_report)

    print(f"Full report saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
