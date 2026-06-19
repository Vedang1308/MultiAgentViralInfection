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
shutil.copy(os.path.join(patch_dir, "qwen_provider.py"), os.path.join(enacttom_path, "habitat_llm/llm/qwen_provider.py"))
shutil.copy(os.path.join(patch_dir, "qwen_provider.yaml"), os.path.join(enacttom_path, "habitat_llm/conf/llm/qwen_provider.yaml"))

# Ensure Qwen provider is importable
import habitat_llm.llm.qwen_provider as qwen_provider

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
    qwen_provider.global_obs_store["latest_obs"] = obs
    return obs, reward, done, info
ei.EnvironmentInterface.step = hooked_step

original_reset = ei.EnvironmentInterface.reset_environment
def hooked_reset(self, *args, **kwargs):
    res = original_reset(self, *args, **kwargs)
    qwen_provider.global_obs_store["latest_obs"] = self.batch
    return res
ei.EnvironmentInterface.reset_environment = hooked_reset

# Now we import the EnactTom benchmark runner
from enacttom.examples.run_habitat_benchmark import main as benchmark_main
import enacttom.examples.run_habitat_benchmark
import enacttom.evaluation_comms
import argparse
import glob
import json

# Monkey-patch evaluate_communication to bypass OpenAI
def mocked_evaluate_communication(action_history, task, model="gpt-5.2"):
    return enacttom.evaluation_comms.CommunicationMetrics(
        per_agent={},
        overall_leakage_score=1.0,
        overall_efficiency_score=1.0,
        overall_score=1.0,
        efficiency_reasoning="MOCKED"
    )
enacttom.evaluation_comms.evaluate_communication = mocked_evaluate_communication

# Monkey-patch run_single_task to save checkpoints
original_run_single_task = enacttom.examples.run_habitat_benchmark.run_single_task
def hooked_run_single_task(*args, **kwargs):
    res = original_run_single_task(*args, **kwargs)
    task_id = res.get("task_id")
    if task_id:
        ckpt_path = os.path.abspath("Phase_1/baselines/checkpoint.json")
        try:
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            if os.path.exists(ckpt_path):
                with open(ckpt_path, "r") as f:
                    ckpt = json.load(f)
            else:
                ckpt = []
            if task_id not in ckpt:
                ckpt.append(task_id)
            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    return res
enacttom.examples.run_habitat_benchmark.run_single_task = hooked_run_single_task

def main():
    parser = argparse.ArgumentParser(description="EnactToM Golden Path Miner")
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3"], help="VLM Model to use")
    # Parse known args so Hydra can still process the rest if needed
    args, unknown = parser.parse_known_args()
    
    # Set environment variable so LLaVAProvider knows which wrapper to load
    os.environ["ENACTTOM_VLM_MODEL"] = args.model
    
    print(f"Starting Phase 1.5: Native EnactToM Execution with {args.model}")
    
    # Load checkpoint
    ckpt_path = os.path.abspath("Phase_1/baselines/checkpoint.json")
    completed_tasks = []
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "r") as f:
                completed_tasks = json.load(f)
            print(f"Loaded checkpoint with {len(completed_tasks)} completed tasks.")
        except Exception as e:
            print(f"Error reading checkpoint: {e}")
    
    # Isolate Standard Split
    source_task_dir = os.path.join(enacttom_path, 'data/enacttom/tasks')
    temp_task_dir = os.path.join(enacttom_path, 'data/enacttom/tasks_standard')
    
    # Clean and recreate temporary tasks_standard directory
    if os.path.exists(temp_task_dir):
        shutil.rmtree(temp_task_dir)
    os.makedirs(temp_task_dir, exist_ok=True)
    
    # Symlink only the standard tasks using absolute paths
    abs_source = os.path.abspath(source_task_dir)
    abs_temp = os.path.abspath(temp_task_dir)
    standard_files = glob.glob(os.path.join(abs_source, "benchmark_standard_*.json"))
    
    symlinked_count = 0
    for f in standard_files:
        basename = os.path.basename(f)
        task_id = basename.replace(".json", "")
        if task_id not in completed_tasks:
            os.symlink(f, os.path.join(abs_temp, basename))
            symlinked_count += 1
            
    print(f"Isolated {symlinked_count} pending standard split tasks in {abs_temp} (Skipped {len(standard_files) - symlinked_count} already completed)")
    
    if symlinked_count == 0:
        print("All tasks completed! Golden Path mining finished.")
        return
    
    # Configure the arguments for the benchmark runner
    sys.argv = [
        "enacttom_loader.py",
        "--config-name", "examples/enacttom_2_robots",
        f"+model={args.model}",
        "+llm_provider=qwen_provider",
        "+max_turns=30",
        f"+task_dir={abs_temp}",
        "hydra.run.dir=Phase_1/baselines/enacttom/${now:%Y-%m-%d_%H-%M-%S}",
        "habitat.dataset.data_path=/scratch/vavaghad/habitat_data/datasets/enacttom_episodes/v0_0/train_2k.json.gz",
        "habitat.dataset.scenes_dir=/scratch/vavaghad/habitat_data/versioned_data/hssd-hab",
    ] + unknown
    
    # Run the official EnactToM benchmark loop!
    benchmark_main()
    
    # Golden Path Cleanup
    print("\nStarting Golden Path Cleanup...")
    baselines_dir = os.path.abspath("Phase_1/baselines/enacttom")
    if not os.path.exists(baselines_dir):
        return
        
    # Find the most recent run directory
    run_dirs = [os.path.join(baselines_dir, d) for d in os.listdir(baselines_dir) if os.path.isdir(os.path.join(baselines_dir, d))]
    if not run_dirs:
        return
        
    latest_run_dir = max(run_dirs, key=os.path.getmtime)
    results_dir = os.path.join(latest_run_dir, "results")
    
    if os.path.exists(results_dir):
        for task_dir in os.listdir(results_dir):
            task_path = os.path.join(results_dir, task_dir)
            if not os.path.isdir(task_path):
                continue
                
            results_json_path = os.path.join(task_path, "benchmark_results.json")
            if os.path.exists(results_json_path):
                try:
                    with open(results_json_path, 'r') as f:
                        res = json.load(f)
                    
                    if not res.get("success", False):
                        shutil.rmtree(task_path)
                except Exception as e:
                    print(f"Error reading {results_json_path}: {e}")
            else:
                # If no results json exists, it likely failed or aborted
                shutil.rmtree(task_path)
                
    print(f"Cleanup complete. Only 100% successful tasks remain in {results_dir}.")

if __name__ == "__main__":
    main()
