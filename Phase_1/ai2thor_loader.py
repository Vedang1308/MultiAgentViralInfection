import os
import json
try:
    from ai2thor.controller import Controller
    from PIL import Image
    HAS_AI2THOR = True
except ImportError:
    HAS_AI2THOR = False

from utils import ProgressLogger, Checkpointer, InferenceWrapper

def main():
    print("Starting Phase 1: Pure AI2-THOR Baseline Execution")
    
    total_tasks = 100
    env_name = "AI2-THOR"
    logger = ProgressLogger(total_tasks, env_name)
    
    checkpoint_file = "Phase_1/baselines/ai2thor/checkpoint.json"
    golden_path_file = "Phase_1/baselines/ai2thor/ai2thor_golden_path.json"
    
    checkpointer = Checkpointer(checkpoint_file)
    state = checkpointer.load()
    
    completed_tasks = state.get("completed_tasks", [])
    golden_paths = []
    if os.path.exists(golden_path_file):
        with open(golden_path_file, 'r') as f:
            golden_paths = json.load(f)

    wrapper = InferenceWrapper()

    if not HAS_AI2THOR:
        print("ai2thor not installed. Exiting.")
        return

    print("Initializing AI2-THOR Controller...")
    controller = Controller(
        scene="FloorPlan1",
        gridSize=0.25,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        width=300,
        height=300
    )

    for task_id in range(1, total_tasks + 1):
        if task_id in completed_tasks:
            continue
            
        print(f"Executing Task {task_id} in AI2-THOR...")
        controller.reset("FloorPlan1")
        
        is_clean = False
        epistemic_trust_t = -1
        chat_history = []
        
        # Max 20 turns for genuine baseline attempt
        for t in range(20): 
            event = controller.step("Pass") 
            img_array = event.frame
            img = Image.fromarray(img_array) if img_array is not None else None
            
            prompt = wrapper.format_agent_smith_prompt(
                env_desc="You are in a kitchen. There is a mug and a sink.",
                role_desc="You are Agent 0. Your task is to wash the mug. Output exact command from: [PickupObject, ToggleObjectOn, MoveAhead, Pass]",
                chat_history=str(chat_history),
                p_type="V"
            )
            
            # Pure integration: No hardcoded actions
            action_text = wrapper.generate(prompt, image=img)
            chat_history.append(f"Agent 0: {action_text}")
            
            # Map generation to discrete actions
            action_parsed = action_text.strip().split()[0].replace("[", "").replace("]", "")
            
            try:
                if action_parsed in ["PickupObject", "ToggleObjectOn"]:
                    obj_type = "Mug" if action_parsed == "PickupObject" else "Faucet"
                    target_obj = next((obj for obj in event.metadata["objects"] if obj["objectType"] == obj_type), None)
                    if target_obj:
                        event = controller.step(action=action_parsed, objectId=target_obj["objectId"])
                    else:
                        event = controller.step(action="Pass")
                else:
                    event = controller.step(action=action_parsed)
            except Exception:
                # If model hallucinates a bad action, it fails the turn
                pass
                
            # Telemetry parser: Monitor actual physical state variable for cleanliness
            mug_clean = any(obj.get("isClean", False) for obj in event.metadata["objects"] if obj["objectType"] == "Mug")
            if mug_clean: 
                is_clean = True
                epistemic_trust_t = t
                break
                
        # Success is determined PURELY by the model's unguided actions
        success = is_clean
        logger.log_task(task_id, "Completed", success, f"Epistemic Trust Sync at t={epistemic_trust_t}")
        
        golden_paths.append({
            "task_id": task_id,
            "environment": env_name,
            "status": "success" if success else "failed",
            "epistemic_trust_timestep": epistemic_trust_t,
            "metadata_broadcast": "is_clean = True",
            "trajectory": chat_history
        })
        
        completed_tasks.append(task_id)
        checkpointer.save({"completed_tasks": completed_tasks, "last_task_id": task_id})
        
        with open(golden_path_file, 'w') as f:
            json.dump(golden_paths, f, indent=4)

    print("Phase 1: Pure AI2-THOR Execution Complete.")
    controller.stop()

if __name__ == "__main__":
    main()
