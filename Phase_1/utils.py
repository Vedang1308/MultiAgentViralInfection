import os
import json
import time
import torch
import re
from typing import Dict, Any, List

class ProgressLogger:
    def __init__(self, total_tasks: int, env_name: str):
        self.total_tasks = total_tasks
        self.env_name = env_name
        self.start_time = time.time()

    def log_task(self, current_task: int, status: str, success: bool, extra_info: str = ""):
        elapsed = time.time() - self.start_time
        success_str = "SUCCESS" if success else "FAILURE"
        color = "\033[92m" if success else "\033[91m"
        reset = "\033[0m"
        print(f"[{self.env_name}] Task {current_task}/{self.total_tasks} | {color}{success_str}{reset} | Status: {status} | Time: {elapsed:.2f}s | {extra_info}")


class Checkpointer:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path

    def load(self) -> Dict[str, Any]:
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'r') as f:
                return json.load(f)
        return {"completed_tasks": [], "last_task_id": 0}

    def save(self, data: Dict[str, Any]):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        with open(self.checkpoint_path, 'w') as f:
            json.dump(data, f, indent=4)


class Gemma3Wrapper:
    def __init__(self):
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            import torch
            print("Loading Gemma-3 12B IT in bfloat16...")
            model_id = "google/gemma-3-12b-it"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, 
                torch_dtype=torch.bfloat16, 
                device_map="auto"
            )
            print("Gemma-3 loaded successfully.")
            self.active = True
        except ImportError as e:
            print(f"Transformers library error: {e}. Running in simulation mode without real model.")
            self.active = False

    def generate(self, prompt: str, image=None) -> str:
        if not self.active:
            return "[Mocked Action Output]"
            
        import torch
        
        prompt_text = prompt.replace("<image>\\n", "").replace("<image>", "")
        
        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        if image is not None:
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to("cuda", torch.bfloat16)
        else:
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            ).to("cuda", torch.bfloat16)

        generate_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text.strip()
