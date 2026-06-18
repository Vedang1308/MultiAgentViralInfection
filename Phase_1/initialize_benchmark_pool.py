import os
import json
import shutil

TEMPLATE_PATH = "Others/EnactTom/enacttom/task_gen/template/sample_tasks.json"
OUTPUT_DIR = "Others/EnactTom/data/enacttom/tasks"

def initialize_pool():
    if not os.path.exists(TEMPLATE_PATH):
        print(f"Error: Template not found at {TEMPLATE_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(TEMPLATE_PATH, "r") as f:
        template_data = json.load(f)

    print(f"Initializing 300 static benchmark tasks in {OUTPUT_DIR}...")
    
    # Generate 150 Standard and 150 Hard
    for split, count, difficulty in [("standard", 150, "standard"), ("hard", 150, "hard")]:
        for i in range(1, count + 1):
            task_copy = dict(template_data)
            task_copy["task_id"] = f"benchmark_{split}_{i}"
            task_copy["difficulty"] = difficulty
            
            output_file = os.path.join(OUTPUT_DIR, f"{task_copy['task_id']}.json")
            with open(output_file, "w") as out_f:
                json.dump(task_copy, out_f, indent=2)

    print(f"Successfully generated 300 static benchmark tasks.")

if __name__ == "__main__":
    initialize_pool()
