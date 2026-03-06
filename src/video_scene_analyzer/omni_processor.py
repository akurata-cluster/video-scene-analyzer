import os
import json
import torch
import torch_xla.core.xla_model as xm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from typing import Dict, Any, List

class OmniProcessor:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = xm.xla_device()
        
        print(f"Loading processor for {self.model_name}...")
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        
        print(f"Loading model {self.model_name} with BitsAndBytes INT8 quantization...")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        
        print("Moving quantized model to TPU...")
        self.model = self.model.to(self.device)
        self.model.eval()

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
        
        # Use Qwen2-VL message format
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
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # Move inputs to TPU device
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            content = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
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
