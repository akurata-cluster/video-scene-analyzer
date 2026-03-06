import argparse
import logging
from .core import VideoAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Analyze videos into text transcripts and event logs.")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--vision-model", default="huihui-ai/Huihui-Qwen3-Omni-30B-A3B-Instruct-abliterated", help="Omni LLM model name")
    parser.add_argument("--context-window", type=int, default=3, help="Number of previous scene descriptions to include as context")
    parser.add_argument("--scene-threshold", type=float, default=27.0, help="PySceneDetect content threshold (default: 27.0)")
    parser.add_argument("--output-transcript", default="transcript.txt", help="Path to save the dialogue transcript")
    parser.add_argument("--output-event-log", default="event_log.txt", help="Path to save the event log")

    args = parser.parse_args()

    analyzer = VideoAnalyzer(
        vision_model=args.vision_model,
        context_window=args.context_window,
        scene_threshold=args.scene_threshold
    )

    analyzer.analyze(
        video_path=args.video_path,
        output_transcript=args.output_transcript,
        output_event_log=args.output_event_log
    )

if __name__ == "__main__":
    main()
