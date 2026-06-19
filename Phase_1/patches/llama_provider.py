from typing import Dict, List, Optional, Tuple, Union
import sys
import os

from omegaconf import DictConfig
from habitat_llm.llm.base_llm import BaseLLM
from PIL import Image
import numpy as np

# Global store for the latest environment observations patched by our loader
global_obs_store = {}

class LlamaProvider(BaseLLM):
    def __init__(self, conf: DictConfig):
        super().__init__(conf)
        
        # Initialize our custom InferenceWrapper from Phase_1
        phase1_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../Phase_1"))
        if phase1_path not in sys.path:
            sys.path.append(phase1_path)
            
        try:
            from utils import Llama32VisionWrapper
            model_env = os.environ.get("ENACTTOM_VLM_MODEL", "llama3.2")
            if model_env == "llama3.2":
                self.wrapper = Llama32VisionWrapper()
            else:
                self.wrapper = Llama32VisionWrapper()
        except ImportError as e:
            print(f"Error loading Llama32VisionWrapper: {e}")
            self.wrapper = None

    def generate(
        self,
        prompt: Union[str, List[Tuple[str, str]]],
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args=None,
    ) -> str:
        prompt_str = str(prompt)
        
        img = None
        # Try to infer which agent this prompt is for to get the right camera frame
        agent_id = "0"
        if "agent_1" in prompt_str or "Agent 1" in prompt_str:
            agent_id = "1"
            
        obs = global_obs_store.get("latest_obs")
        if obs is not None:
            # EnactToM configures the agent's sensors, e.g. agent_0_head_rgb or agent_0_articulated_agent_arm_rgb
            cam_key = f"agent_{agent_id}_head_rgb"
            if cam_key in obs:
                rgb_tensor = obs[cam_key]
                # Assuming (H, W, C) from habitat
                if hasattr(rgb_tensor, 'cpu'):
                    rgb_array = rgb_tensor.cpu().numpy()
                else:
                    rgb_array = np.array(rgb_tensor)
                
                # Convert to PIL Image
                if len(rgb_array.shape) == 3 and rgb_array.shape[-1] >= 3:
                    img = Image.fromarray(rgb_array[..., :3].astype(np.uint8))
        
        if self.wrapper is not None:
            if img is not None and "<image>" not in prompt_str:
                prompt_str = "<image>\n" + prompt_str
                
            # Fallback if no image found, we still send the prompt
            response = self.wrapper.generate(prompt_str, image=img)
            # LLaVA 1.5 often escapes underscores in Markdown style (e.g. Agent\_0\_Action)
            response = response.replace('\\_', '_')
            return response
        else:
            return "Action: Communicate[\"Hardware Error: LLaVA not initialized\"]"
