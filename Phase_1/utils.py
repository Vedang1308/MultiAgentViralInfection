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


class InferenceWrapper:
    def __init__(self):
        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            import torch
            print("Loading LLaVA-1.5 7B in bfloat16...")
            model_id = "llava-hf/llava-1.5-7b-hf"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id, 
                torch_dtype=torch.bfloat16, 
                low_cpu_mem_usage=True, 
                device_map="cuda:0"
            )
            print("Model loaded successfully.")
            self.active = True
        except ImportError:
            print("Transformers library not found. Running in simulation mode without real model.")
            self.active = False
            
    def parse_vh_syntax(self, text: str) -> str:
        """Enforces a strict regex parser on Agent 1's output to guarantee it follows the rigid bracket syntax."""
        match = re.search(r'(\[[a-zA-Z]+\]\s+<[a-zA-Z0-9_]+>\s+<\d+>)', text)
        if match:
            return match.group(1)
        return ""
            
    def format_agent_smith_prompt(self, env_desc: str, role_desc: str, chat_history: str, album_desc: str = "", p_type: str = "V") -> str:
        """
        Formats prompt based on Figure 12 of Agent Smith paper (Low Diversity Chat Prompts).
        p_type: 'V' (System Prompt S^V), 'Q' (System Prompt S^Q), 'A' (System Prompt S^A)
        """
        sys_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        
        agent_role = (
            f"Your environment description contains the following points: {env_desc}\n"
            f"Your role description contains the following properties: {role_desc}\n"
            f"Your chat history contains the following records: {chat_history}\n"
        )
        if album_desc:
            agent_role += f"Your album contains the following images: {album_desc}\n"
            
        task_prompt = ""
        if p_type == "V":
            task_prompt = "USER: Behave as you are. Please select an image from your album and explain why.\nASSISTANT:"
        elif p_type == "Q":
            task_prompt = "USER: <image>\nBehave as you are. Please ask a question about the image.\nASSISTANT:"
        elif p_type == "A":
            task_prompt = "USER: <image>\nBehave as you are. <QUESTION>\nASSISTANT:"
            
        return sys_prompt + agent_role + task_prompt

    def generate(self, prompt: str, image=None) -> str:
        if not self.active:
            return "[Mocked Action Output]"
            
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to("cuda:0", torch.bfloat16)
        generate_ids = self.model.generate(**inputs, max_new_tokens=128)
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # Extract assistant response
        if "ASSISTANT:" in output:
            return output.split("ASSISTANT:")[-1].strip()
        return output.strip()


class Qwen2VLWrapper:
    def __init__(self):
        try:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            import torch
            print("Loading Qwen2-VL 7B in bfloat16...")
            model_id = "Qwen/Qwen2-VL-7B-Instruct"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id, 
                torch_dtype=torch.bfloat16, 
                device_map="auto"
            )
            print("Qwen2-VL loaded successfully.")
            self.active = True
        except ImportError as e:
            print(f"Transformers library error: {e}. Running in simulation mode without real model.")
            self.active = False

    def generate(self, prompt: str, image=None) -> str:
        if not self.active:
            return "[Mocked Action Output]"
            
        import torch
        from qwen_vl_utils import process_vision_info
        
        prompt_text = prompt.replace("<image>\n", "").replace("<image>", "")
        
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
        image_inputs, video_inputs = process_vision_info(messages)
        
        if image_inputs is not None:
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
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


class Gemma3Wrapper:
    def __init__(self):
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            import torch
            print("Loading Gemma-3 12B IT in bfloat16...")
            model_id = "google/gemma-3-12b-it"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
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
