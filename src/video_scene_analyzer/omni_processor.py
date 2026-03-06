import base64
import json
import os
from openai import OpenAI
from typing import Dict, Any, List

class OmniProcessor:
    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key or "dummy")
        self.model_name = model_name

    def process_chunk(self, chunk_path: str, context: List[str]) -> Dict[str, Any]:
        """
        Sends the physical chunk to the Omni model and asks for BOTH dialogue transcription 
        and visual event log in a structured JSON response.
        """
        # Read the raw chunk file and base64 encode it for the OpenAI API
        with open(chunk_path, "rb") as f:
            video_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Determine MIME type based on extension
        ext = os.path.splitext(chunk_path)[1].lower().strip(".")
        if ext == "mp4":
            mime_type = "video/mp4"
        elif ext in ["mkv", "avi", "mov"]:
            mime_type = f"video/{ext}"
        else:
            mime_type = "video/mp4" # fallback
            
        system_prompt = (
            "You are a natively multimodal Omni model. Your task is to analyze a video chunk and output BOTH "
            "a dialogue transcription and a visual event log for that chunk in a single JSON structured response.\n\n"
            "Respond strictly with a JSON object containing two keys: 'transcription' (string) and 'event_log' (string)."
        )
        
        prompt = "Analyze the provided video chunk."
        if context:
            prompt += "\n\nFor context, here are the descriptions of the previous scenes:\n"
            for i, past_scene in enumerate(context, 1):
                prompt += f"Scene -{len(context) - i + 1}: {past_scene}\n"
        
        prompt += "\nOutput the JSON with 'transcription' and 'event_log'."
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:{mime_type};base64,{video_data}"}
                    }
                ]
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={ "type": "json_object" },
                max_tokens=2048
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except Exception as e:
            return {
                "transcription": f"[Error generating transcription: {e}]",
                "event_log": f"[Error generating event log: {e}]"
            }
