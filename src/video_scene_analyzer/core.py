import json
import logging
from typing import Dict, List, Any
import time

from .scene_processor import detect_scenes
from .audio_processor import AudioProcessor
from .vision_processor import VisionProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoAnalyzer:
    def __init__(
        self,
        whisper_model: str = "base",
        vision_base_url: str = "http://localhost:8000/v1",
        vision_api_key: str = "dummy",
        vision_model: str = "huihui-ai/Huihui-Qwen3-Omni-30B-A3B-Instruct-abliterated",
        context_window: int = 3,
        scene_threshold: float = 27.0
    ):
        self.context_window = context_window
        self.scene_threshold = scene_threshold
        
        logger.info(f"Initializing AudioProcessor with model: {whisper_model}")
        self.audio_proc = AudioProcessor(model_size=whisper_model)
        
        logger.info(f"Initializing VisionProcessor with model: {vision_model} at {vision_base_url}")
        self.vision_proc = VisionProcessor(
            base_url=vision_base_url,
            api_key=vision_api_key,
            model_name=vision_model
        )

    def analyze(self, video_path: str, output_transcript: str, output_event_log: str):
        """Main analysis loop."""
        logger.info(f"Starting analysis for video: {video_path}")
        
        # 1. Detect Scenes
        logger.info("Detecting scenes...")
        scenes = detect_scenes(video_path, self.scene_threshold)
        logger.info(f"Detected {len(scenes)} scenes.")

        # 2. Process Scenes
        transcript_entries = []
        event_entries = []
        vision_context = []

        for idx, (start_time, end_time) in enumerate(scenes):
            logger.info(f"Processing scene {idx + 1}/{len(scenes)} [{start_time:.2f}s - {end_time:.2f}s]")
            
            # Audio Processing
            transcript_text = self.audio_proc.transcribe_chunk(video_path, start_time, end_time)
            
            # Vision Processing
            # Send the previous N descriptions as context to the vision model
            scene_desc = self.vision_proc.describe_scene(
                video_path, start_time, end_time, context=vision_context
            )
            
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
