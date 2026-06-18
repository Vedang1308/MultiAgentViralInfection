import os
import sys

# Add EnactTom to path so we can import its modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
enacttom_path = os.path.join(project_root, "Others", "EnactTom")
if enacttom_path not in sys.path:
    sys.path.insert(0, enacttom_path)
import shutil

# Automatically copy patches into EnactToM so Hydra and Python can find them
patch_dir = os.path.join(os.path.dirname(__file__), "patches")
shutil.copy(os.path.join(patch_dir, "llava_local.py"), os.path.join(enacttom_path, "habitat_llm/llm/llava_local.py"))
shutil.copy(os.path.join(patch_dir, "llava_local.yaml"), os.path.join(enacttom_path, "habitat_llm/conf/llm/llava_local.yaml"))

# Ensure LLaVA local provider is importable
import habitat_llm.llm.llava_local as llava_local

# Ensure the 'data' directory exists in the CWD because habitat-lab expects to write 'data/default.physics_config.json'
os.makedirs("data", exist_ok=True)
import getpass
user = getpass.getuser()
scratch_hssd = f"/scratch/{user}/habitat_data/versioned_data/hssd-hab"
try:
    if not os.path.exists("data/hssd-hab"):
        os.symlink(scratch_hssd, "data/hssd-hab")
    
    # Newer HSSD datasets flatten metadata into the root.
    # Create the metadata folder and symlink the required files specifically.
    meta_dir = f"{scratch_hssd}/metadata"
    os.makedirs(meta_dir, exist_ok=True)
    for file in ["object_categories_filtered.csv", "room_objects.json", "fpmodels-with-decomposed.csv", "affordance_objects.csv"]:
        if not os.path.exists(f"{meta_dir}/{file}") and os.path.exists(f"{scratch_hssd}/{file}"):
            os.symlink(f"../{file}", f"{meta_dir}/{file}")
except OSError as e:
    print(f"Warning: Failed to setup dataset symlinks: {e}")

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
        "--config-name", "examples/enacttom_2_robots",
        "+model=llava-1.5",
        "+llm_provider=llava_local",
        "+max_turns=30",
        f"+task_dir={os.path.join(enacttom_path, 'data/enacttom/tasks')}",
        "hydra.run.dir=Phase_1/baselines/enacttom/${now:%Y-%m-%d_%H-%M-%S}",
        "habitat.dataset.data_path=/scratch/vavaghad/habitat_data/datasets/enacttom_episodes/v0_0/train_2k.json.gz",
        "habitat.dataset.scenes_dir=/scratch/vavaghad/habitat_data/versioned_data/hssd-hab",
        # Default single-agent or multi-agent is handled by the task JSON
    ]
    
    # Run the official EnactToM benchmark loop!
    # This will automatically load the tasks, initialize DecPOMDPEnv,
    # and call our LLaVAProvider for planning.
    benchmark_main()

if __name__ == "__main__":
    main()
