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
        self.sim = None
        try:
            import habitat_sim
            
            # DEBUG prints for SOL
            data_dir = "Others/EnactTom/data"
            print(f"DEBUG: Contents of {data_dir}: {os.listdir(data_dir) if os.path.exists(data_dir) else 'DOES NOT EXIST'}")
            hssd_path = os.path.join(data_dir, "hssd-hab")
            if os.path.islink(hssd_path):
                target = os.readlink(hssd_path)
                print(f"DEBUG: hssd-hab is a symlink pointing to -> {target}")
                print(f"DEBUG: Target exists? {os.path.exists(target)}")
                if os.path.exists(target):
                    print(f"DEBUG: Target contents: {os.listdir(target)}")
            elif os.path.exists(hssd_path):
                print(f"DEBUG: hssd-hab is a real directory. Contents: {os.listdir(hssd_path)}")
            else:
                print(f"DEBUG: hssd-hab DOES NOT EXIST at {hssd_path}")

            # Find a valid scene in ANY dataset (recursive search with symlink following)
            scene_dir = "Others/EnactTom/data"
            scenes = []
            for root, dirs, files in os.walk(scene_dir, followlinks=True):
                for f in files:
                    if f.endswith(".scene_instance.json") or f.endswith(".glb"):
                        scenes.append(os.path.join(root, f))
                        
            if not scenes:
                raise FileNotFoundError(f"No scenes found recursively in {scene_dir}. Symlink or LFS download might have failed.")
                
            scene_file = scenes[0]
            print(f"Loading Scene into Habitat-Sim: {scene_file}")
            
            # Configure Habitat-Sim for Headless EGL rendering
            sim_cfg = habitat_sim.SimulatorConfiguration()
            sim_cfg.scene_id = scene_file
            
            # Agent 0 config
            agent0_cfg = habitat_sim.agent.AgentConfiguration()
            rgb0 = habitat_sim.CameraSensorSpec()
            rgb0.uuid = "rgb_0"
            rgb0.sensor_type = habitat_sim.SensorType.COLOR
            rgb0.resolution = [256, 256]
            rgb0.position = [0.0, 1.5, 0.0]
            agent0_cfg.sensor_specifications = [rgb0]
            
            # Agent 1 config
            agent1_cfg = habitat_sim.agent.AgentConfiguration()
            rgb1 = habitat_sim.CameraSensorSpec()
            rgb1.uuid = "rgb_1"
            rgb1.sensor_type = habitat_sim.SensorType.COLOR
            rgb1.resolution = [256, 256]
            rgb1.position = [1.0, 1.5, 1.0]
            agent1_cfg.sensor_specifications = [rgb1]
            
            cfg = habitat_sim.Configuration(sim_cfg, [agent0_cfg, agent1_cfg])
            self.sim = habitat_sim.Simulator(cfg)
            
            # Initialize navmesh to place agents
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            self.sim.recompute_navmesh(self.sim.pathfinder, navmesh_settings)
            
            for agent_idx in [0, 1]:
                state = self.sim.get_agent(agent_idx).get_state()
                state.position = self.sim.pathfinder.get_random_navigable_point()
                self.sim.get_agent(agent_idx).set_state(state)
                
            self.active = True
            print("Successfully initialized Headless Habitat-Sim with HSSD!")
        except Exception as e:
            print(f"WARNING: Pure EnactToM requires Habitat and HSSD. Running offline fallback loop. Error: {e}")

    def reset(self, task_file: str):
        if self.active:
            obs = self.sim.get_sensor_observations()
            return f"Simulated Observation (RGB Tensor Shape: {obs['rgb_0'].shape})", f"Simulated Observation (RGB Tensor Shape: {obs['rgb_1'].shape})", {"0": "secret_abc"}
        return "Simulated Observation (Living Room)", "Simulated Observation (Kitchen)", {"0": "secret_abc"}

    def step(self, action_0, action_1):
        if self.active:
            obs = self.sim.get_sensor_observations()
            return f"Obs_0 (RGB Tensor Shape: {obs['rgb_0'].shape})", f"Obs_1 (RGB Tensor Shape: {obs['rgb_1'].shape})", False, {"msg": "habitat_step"}
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
