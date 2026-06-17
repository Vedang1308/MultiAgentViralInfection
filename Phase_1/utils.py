import os
import json
import time
from typing import Dict, Any

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


def load_llava_model():
    """
    Loads LLaVA-1.5 7B in bfloat16 to maximize A100 memory headroom.
    HF_HOME must be set to the scratch directory in the environment.
    """
    try:
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        import torch
        print("Loading LLaVA-1.5 7B in bfloat16...")
        model_id = "llava-hf/llava-1.5-7b-hf"
        processor = AutoProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            device_map="cuda:0"
        )
        print("Model loaded successfully.")
        return processor, model
    except ImportError:
        print("Transformers library not found. Running in simulation mode without real model.")
        return None, None
