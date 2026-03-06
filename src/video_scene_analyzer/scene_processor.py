import os
import tempfile
from typing import List, Tuple
from scenedetect import detect, ContentDetector, SceneManager, open_video
from scenedetect.video_splitter import split_video_ffmpeg

def detect_scenes(video_path: str, threshold: float = 27.0) -> List[Tuple[float, float, str]]:
    """
    Detects scenes in a video using ContentDetector and splits them into physical chunks.
    Returns a list of tuples (start_time_sec, end_time_sec, chunk_file_path).
    """
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    
    if not scene_list:
        # If no scenes detected, treat the whole video as one scene
        scene_list = [(video.base_timecode, video.duration)]
        
    temp_dir = tempfile.mkdtemp(prefix="scene_chunks_")
    output_template = os.path.join(temp_dir, "chunk_$SCENE_NUMBER.mp4")
    
    # Use split_video_ffmpeg to cut the video
    split_video_ffmpeg(video_path, scene_list, output_file_template=output_template, show_progress=False)
    
    scenes_info = []
    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        chunk_path = os.path.join(temp_dir, f"chunk_{i+1:03d}.mp4")
        scenes_info.append((start_time, end_time, chunk_path))
        
    return scenes_info
