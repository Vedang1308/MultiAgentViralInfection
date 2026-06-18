import os
import sys

# Add EnactTom to path so we can import its modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
enacttom_path = os.path.join(project_root, "Others", "EnactTom")
if enacttom_path not in sys.path:
    sys.path.insert(0, enacttom_path)

# Ensure LLaVA local provider is importable
import habitat_llm.llm.llava_local as llava_local

# Patch EnvironmentInterface to capture visual observations for LLaVA
import habitat_llm.agent.env.environment_interface as ei

original_step = ei.EnvironmentInterface.step
def hooked_step(self, low_level_actions):
    obs, reward, done, info = original_step(self, low_level_actions)
    llava_local.global_obs_store["latest_obs"] = obs
    return obs, reward, done, info
ei.EnvironmentInterface.step = hooked_step

original_reset = ei.EnvironmentInterface.reset_environment
def hooked_reset(self, *args, **kwargs):
    res = original_reset(self, *args, **kwargs)
    llava_local.global_obs_store["latest_obs"] = self.batch
    return res
ei.EnvironmentInterface.reset_environment = hooked_reset

# Now we import the EnactTom benchmark runner
from enacttom.examples.run_habitat_benchmark import main as benchmark_main

def main():
    print("Starting Phase 1.5: Native EnactToM Execution with LLaVA-1.5")
    
    # Configure the arguments for the benchmark runner
    # We specify our custom LLaVA provider and task directory
    sys.argv = [
        "enacttom_loader.py",
        "+model=llava-1.5",
        "+llm_provider=llava_local",
        "task_dir=data/enacttom/tasks",
        # Default single-agent or multi-agent is handled by the task JSON
    ]
    
    # Run the official EnactToM benchmark loop!
    # This will automatically load the tasks, initialize DecPOMDPEnv,
    # and call our LLaVAProvider for planning.
    benchmark_main()

if __name__ == "__main__":
    main()
