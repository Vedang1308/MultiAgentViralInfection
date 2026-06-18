import gzip
import json
import os
import copy
import getpass

user = getpass.getuser()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
enacttom_path = os.path.join(project_root, "Others", "EnactTom")

EPISODES_FILE = f"/scratch/{user}/habitat_data/datasets/enacttom_episodes/v0_0/train_2k.json.gz"
TASKS_DIR = os.path.join(enacttom_path, "data/enacttom/tasks")

def extract_tasks():
    os.makedirs(TASKS_DIR, exist_ok=True)
    
    print(f"Loading episodes from {EPISODES_FILE}...")
    with gzip.open(EPISODES_FILE, 'rt') as f:
        data = json.load(f)
        
    episodes = data.get('episodes', [])
    if len(episodes) < 300:
        print(f"Warning: Only {len(episodes)} episodes found in train_2k.json.gz.")
    
    print("Extracting 150 Standard and 150 Hard tasks...")
    for i in range(min(300, len(episodes))):
        ep = episodes[i]
        
        split = "standard" if i < 150 else "hard"
        local_idx = i if i < 150 else (i - 150)
        
        task_name = f"benchmark_{split}_{local_idx}"
        
        goal_literals = []
        objects_mentioned = set()
        
        for prop in ep.get('evaluation_propositions', []):
            func = prop.get('function_name')
            args = prop.get('args', {})
            
            obj_handles = args.get('object_handles', [])
            rec_handles = args.get('receptacle_handles', [])
            
            if func in ["is_on_top", "is_inside", "is_next_to", "is_on_floor", "is_in_room"]:
                if obj_handles and rec_handles:
                    for obj in obj_handles:
                        for rec in rec_handles:
                            goal_literals.append(f"({func} {obj} {rec})")
                            objects_mentioned.add(obj)
                            objects_mentioned.add(rec)
            elif func in ["is_filled", "is_empty", "is_clean", "is_powered_on", "is_powered_off"]:
                if obj_handles:
                    for obj in obj_handles:
                        goal_literals.append(f"({func} {obj})")
                        objects_mentioned.add(obj)
        
        if not goal_literals:
            goal_literals.append("(and)")
            
        goal_pddl = "(and " + " ".join(goal_literals) + ")" if len(goal_literals) > 1 else goal_literals[0]
        
        objects_pddl = " ".join(objects_mentioned) + " - object agent_0 agent_1 - agent"
        
        owner_pddl_lines = []
        # Alternate owners to create epistemic dependencies (Theory of Mind requirements)
        for j, lit in enumerate(goal_literals):
            owner = "agent_0" if j % 2 == 0 else "agent_1"
            owner_pddl_lines.append(f"    ({owner} {lit})")
            
        owner_pddl = "\n".join(owner_pddl_lines)
        
        problem_pddl = f"""(define (problem {task_name})
  (:domain enacttom)
  (:objects {objects_pddl})
  (:init
    (agent_in_room agent_0 room_0)
    (agent_in_room agent_1 room_0)
  )
  (:goal {goal_pddl})
  (:goal-owners
{owner_pddl}
  )
)"""

        new_ep = copy.deepcopy(ep)
        new_ep['problem_pddl'] = problem_pddl
        new_ep['task_id'] = task_name
        new_ep['title'] = task_name
        new_ep['task'] = ep.get('instruction', "No instruction provided.")
        new_ep['category'] = "cooperative"
        new_ep['num_agents'] = 2
        new_ep['agent_actions'] = {
            "agent_0": ['Navigate', 'Pick', 'Place', 'Open', 'Close', 'Rearrange', 'Wait', 'Communicate', 'FindReceptacleTool', 'FindObjectTool', 'FindAgentActionTool', 'FindRoomTool'],
            "agent_1": ['Navigate', 'Pick', 'Place', 'Open', 'Close', 'Rearrange', 'Wait', 'Communicate', 'FindReceptacleTool', 'FindObjectTool', 'FindAgentActionTool', 'FindRoomTool']
        }
        
        # Remove legacy fields to satisfy EnactToM parser strict checks
        if 'evaluation_propositions' in new_ep:
            del new_ep['evaluation_propositions']
        if 'evaluation_constraints' in new_ep:
            del new_ep['evaluation_constraints']
        if 'evaluation_proposition_dependencies' in new_ep:
            del new_ep['evaluation_proposition_dependencies']
            
        out_path = os.path.join(TASKS_DIR, f"{task_name}.json")
        with open(out_path, 'w') as out_f:
            json.dump(new_ep, out_f, indent=2)
            
    print(f"Successfully extracted and compiled 300 tasks into {TASKS_DIR}/")

if __name__ == "__main__":
    extract_tasks()
