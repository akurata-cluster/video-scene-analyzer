import base64
import os
import tempfile
from typing import List
import cv2
from openai import OpenAI

class VisionProcessor:
    def __init__(self, base_url: str, api_key: str, model_name: str, n_frames: int = 3):
        """Initialize the vision processor connecting to an OpenAI-compatible endpoint."""
        self.client = OpenAI(base_url=base_url, api_key=api_key or "dummy")
        self.model_name = model_name
        self.n_frames = n_frames

    def extract_frames(self, video_path: str, start_time: float, end_time: float) -> List[str]:
        """Extracts n_frames evenly spaced from the video scene, returns base64 strings."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if not fps or fps <= 0:
            fps = 24.0 # Fallback
            
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Protect against weird duration
        if end_frame <= start_frame:
            end_frame = start_frame + 1

        # Calculate evenly spaced frame indices
        step = max(1, (end_frame - start_frame) // (self.n_frames + 1))
        frame_indices = [start_frame + step * i for i in range(1, self.n_frames + 1)]
        
        base64_frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame to avoid massive payloads
            max_dim = 1024
            h, w = frame.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / float(max(h, w))
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            base64_img = base64.b64encode(buffer).decode('utf-8')
            base64_frames.append(base64_img)

        cap.release()
        return base64_frames

    def describe_scene(self, video_path: str, start_time: float, end_time: float, context: List[str]) -> str:
        """Asks the Vision LLM to describe what is happening in the scene frames."""
        base64_frames = self.extract_frames(video_path, start_time, end_time)
        if not base64_frames:
            return "No visual content could be extracted for this scene."

        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": "You are a helpful video scene describer. You accurately describe the actions, people, and environment in the provided frames. You maintain continuity with previous scenes."
            }
        ]
        
        prompt = "Describe the events occurring in these sequential frames from the current video scene. Focus on actions, changes, and key subjects.\n"
        if context:
            prompt += "\nFor context, here are the descriptions of the previous scenes:\n"
            for i, past_scene in enumerate(context, 1):
                prompt += f"Scene -{len(context) - i + 1}: {past_scene}\n"
            prompt += "\nNow describe the current scene:\n"

        content = [{"type": "text", "text": prompt}]
        for frame in base64_frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
            })

        messages.append({
            "role": "user",
            "content": content
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Error generating vision description: {e}]"
