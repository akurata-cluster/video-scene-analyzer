import os
import json
from typing import Dict, Any, List
from vllm import LLM, SamplingParams

class OmniProcessor:
    def __init__(self, model_name: str):
        self.model_name = model_name
        
        print(f"Loading model {self.model_name} with vLLM...")
        self.model = LLM(
            model=self.model_name,
            trust_remote_code=True,
            limit_mm_per_prompt={"video": 1}
        )
        self.sampling_params = SamplingParams(max_tokens=2048, temperature=0.0)

    def process_chunk(self, chunk_path: str, context: List[str]) -> Dict[str, Any]:
        """
        Sends the physical chunk to the local Omni model and asks for BOTH dialogue transcription 
        and visual event log in a structured JSON response.
        """
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
        
        # Use Qwen2-VL message format supported natively by vLLM Chat Completion API
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": chunk_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        try:
            outputs = self.model.chat(
                messages=messages,
                sampling_params=self.sampling_params
            )
            
            content = outputs[0].outputs[0].text
            
            # Clean up JSON
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            return json.loads(content)
        except Exception as e:
            return {
                "transcription": f"[Error generating transcription: {e}]",
                "event_log": f"[Error generating event log: {e}]"
            }
