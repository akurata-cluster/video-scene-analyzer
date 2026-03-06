import json
import logging
import os
from typing import Dict, List, Any

from .scene_processor import detect_scenes
from .omni_processor import OmniProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoAnalyzer:
    def __init__(
        self,
        vision_model: str = "huihui-ai/Huihui-Qwen3-Omni-30B-A3B-Instruct-abliterated",
        context_window: int = 3,
        scene_threshold: float = 27.0
    ):
        self.context_window = context_window
        self.scene_threshold = scene_threshold
        
        logger.info(f"Initializing OmniProcessor with model: {vision_model}")
        self.omni_proc = OmniProcessor(
            model_name=vision_model
        )

    def analyze(self, video_path: str, output_transcript: str, output_event_log: str):
        """Main analysis loop."""
        logger.info(f"Starting analysis for video: {video_path}")
        
        # 1. Detect Scenes & Cut Chunks
        logger.info("Detecting scenes and cutting chunks...")
        scenes = detect_scenes(video_path, self.scene_threshold)
        logger.info(f"Detected {len(scenes)} scenes.")

        # 2. Process Scenes
        transcript_entries = []
        event_entries = []
        vision_context = []

        for idx, (start_time, end_time, chunk_path) in enumerate(scenes):
            logger.info(f"Processing scene {idx + 1}/{len(scenes)} [{start_time:.2f}s - {end_time:.2f}s]")
            
            # Send chunk to Omni model
            result = self.omni_proc.process_chunk(chunk_path, context=vision_context)
            
            # Clean up the physical chunk
            if os.path.exists(chunk_path):
                try:
                    os.remove(chunk_path)
                except Exception as e:
                    logger.warning(f"Could not remove chunk file {chunk_path}: {e}")
            
            transcript_text = result.get("transcription", "")
            scene_desc = result.get("event_log", "")
            
            # Update context
            vision_context.append(scene_desc)
            if len(vision_context) > self.context_window:
                vision_context.pop(0)

            # Record
            if transcript_text:
                transcript_entries.append({
                    "scene": idx + 1,
                    "start": start_time,
                    "end": end_time,
                    "text": transcript_text
                })
            
            event_entries.append({
                "scene": idx + 1,
                "start": start_time,
                "end": end_time,
                "description": scene_desc
            })

        # 3. Save Outputs
        logger.info(f"Saving outputs to {output_transcript} and {output_event_log}")
        with open(output_transcript, "w", encoding="utf-8") as f:
            for entry in transcript_entries:
                f.write(f"[{entry['start']:.2f}s - {entry['end']:.2f}s] {entry['text']}\n")

        with open(output_event_log, "w", encoding="utf-8") as f:
            for entry in event_entries:
                f.write(f"[{entry['start']:.2f}s - {entry['end']:.2f}s] {entry['description']}\n")

        logger.info("Analysis complete.")
