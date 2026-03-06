import os
from typing import List, Tuple
from scenedetect import detect, ContentDetector, SceneManager, open_video
from scenedetect.frame_timecode import FrameTimecode

def detect_scenes(video_path: str, threshold: float = 27.0) -> List[Tuple[float, float]]:
    """
    Detects scenes in a video using ContentDetector.
    Returns a list of tuples (start_time_sec, end_time_sec).
    """
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    
    scenes_sec = []
    if not scene_list:
        # If no scenes detected, treat the whole video as one scene
        scenes_sec.append((0.0, video.duration.get_seconds()))
        return scenes_sec
        
    for scene in scene_list:
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        scenes_sec.append((start_time, end_time))
        
    return scenes_sec
