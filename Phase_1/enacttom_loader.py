import os
import json
import glob
from typing import Dict, List, Any
from utils import ProgressLogger, Checkpointer, InferenceWrapper

class PureEnactToMEnv:
    """
    Direct interface to the EnactToM Benchmark and Habitat Simulator.
    Requires EnactToM repository and HSSD dataset installed.
    """
    def __init__(self):
        self.active = False
        try:
            import habitat
            import habitat_sim
            from enacttom.envs import DecPOMDPEnv
            self.env = DecPOMDPEnv(dataset="HSSD")
            self.active = True
        except ImportError:
            print("WARNING: Pure EnactToM requires Habitat and HSSD. Running offline fallback loop.")

    def reset(self, task_file: str):
        if self.active:
            return self.env.reset(task_file)
        return "Simulated Observation (Living Room)", "Simulated Observation (Kitchen)", {"0": "secret_abc"}

    def step(self, action_0, action_1):
        if self.active:
            return self.env.step([action_0, action_1])
        return "Obs_0", "Obs_1", False, {"msg": "offline_step"}

def main():
    print("Starting Phase 1: Pure EnactToM Baseline Execution (HSSD Dataset)")
    
    # Load actual tasks from the EnactToM dataset directory
    task_dir = "Others/EnactTom/data/enacttom/tasks"
    task_files = glob.glob(f"{task_dir}/*.json")
    total_tasks = len(task_files) if task_files else 100
    
    env_name = "EnactToM"
    logger = ProgressLogger(total_tasks, env_name)
    
    checkpoint_file = "Phase_1/baselines/enacttom/checkpoint.json"
    golden_path_file = "Phase_1/baselines/enacttom/enacttom_golden_path.json"
    
    checkpointer = Checkpointer(checkpoint_file)
    state = checkpointer.load()
    
    completed_tasks = state.get("completed_tasks", [])
    golden_paths = []
    if os.path.exists(golden_path_file):
        with open(golden_path_file, 'r') as f:
            golden_paths = json.load(f)

    wrapper = InferenceWrapper()
    env = PureEnactToMEnv()

    for task_id in range(1, total_tasks + 1):
        if task_id in completed_tasks:
            continue
            
        task_file = task_files[task_id - 1] if task_files else f"task_{task_id}.json"
        print(f"Executing Task {task_id} in EnactToM ({task_file})...")
        
        obs_0, obs_1, private_secrets = env.reset(task_file)
        
        epistemic_trust_t = -1
        success = False
        chat_history = []
        
        for t in range(5):
            # Agent 0 generates message based on true visual observation and secret
            env_desc_0 = f"{obs_0}. Private Secret: {private_secrets.get('0', '')}"
            sys_prompt_0 = wrapper.format_agent_smith_prompt(
                env_desc=env_desc_0,
                role_desc="You are Agent 0. Communicate the secret effectively.",
                chat_history=str(chat_history),
                p_type="Q"
            )
            
            msg_0 = wrapper.generate(sys_prompt_0)
            chat_history.append(f"Agent 0: {msg_0}")
            
            # Agent 1 processes message and decides action
            env_desc_1 = f"{obs_1}."
            sys_prompt_1 = wrapper.format_agent_smith_prompt(
                env_desc=env_desc_1,
                role_desc="You are Agent 1. Take action based on the chat history.",
                chat_history=str(chat_history),
                p_type="A"
            )
            
            action_1 = wrapper.generate(sys_prompt_1)
            chat_history.append(f"Agent 1: {action_1}")
            
            obs_0, obs_1, done, info = env.step("Pass", action_1)
            
            if done:
                # The pure environment signals success
                epistemic_trust_t = t
                success = True
                break
                
        logger.log_task(task_id, "Completed", success, f"Epistemic Trust Sync at t={epistemic_trust_t}")
        
        golden_paths.append({
            "task_id": task_id,
            "environment": env_name,
            "status": "success" if success else "failed",
            "epistemic_trust_timestep": epistemic_trust_t,
            "trajectory": chat_history
        })
        
        completed_tasks.append(task_id)
        checkpointer.save({"completed_tasks": completed_tasks, "last_task_id": task_id})
        
        with open(golden_path_file, 'w') as f:
            json.dump(golden_paths, f, indent=4)

    print("Phase 1: Pure EnactToM Execution Complete.")

if __name__ == "__main__":
    main()
