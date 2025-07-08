import argparse
import os
import yaml
from collections import defaultdict
import numpy as np
import pickle

def process_trials(project_dir):
    
    config_path = os.path.join(project_dir, "config.yaml")
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    
    num_tasks = config["env_config"]["num_tasks"]
    num_trials = config["exp_config"]["num_trials"]
   
    eval_info = {}
    for task in range(num_tasks):
        eval_info["task_" + str(task)] = {"success":[], "success_final": [], "time": []}
        

    max_tasks = []
    for trial_idx in range(num_trials):
        trial_info_path = os.path.join(project_dir, f"trial_{trial_idx}", "data", "eval", "info.yml")
        if os.path.exists(trial_info_path):
            with open(trial_info_path, "r") as info_file:
                trial_info = yaml.safe_load(info_file)
           
                max_task = 0 

                for task in range(num_tasks):
                    if "task_" + str(task) in trial_info.keys():
                        eval_info["task_" + str(task)]["success"].append(trial_info["task_" + str(task)]["success"][-1])
                        
                    if "task_" + str(task) + "_final_policy" in trial_info.keys():
                        eval_info["task_" + str(task)]["success_final"].append(trial_info["task_" + str(task)+ "_final_policy"]["success"][-1])
                        
                        
                    # load time info
                    trial_time_path = os.path.join(project_dir, f"trial_{trial_idx}", "data", "train", "checkpoints", "params_task_" + str(task) + ".pkl")
                    
                    if os.path.exists(trial_time_path):
                        with open(trial_time_path, "rb") as time_file:
                            time, _ = pickle.load(time_file)
                            eval_info["task_" + str(task)]["time"].append(time)
                            
                            max_task = task
                max_tasks.append(max_task)
                        
                        

    """
    for task in range(num_tasks):
        task_key = "task_" + str(task)
        eval_info[task_key]["success"] = {
            "mean": float(np.mean(eval_info[task_key]["success"])),
            "std": float(np.std(eval_info[task_key]["success"]))
        }
        eval_info[task_key]["success_final"] = {
            "mean": float(np.mean(eval_info[task_key]["success_final"])),
            "std": float(np.std(eval_info[task_key]["success_final"]))
        }
        
        eval_info[task_key]["time"] = {
            "mean": float(np.mean(eval_info[task_key]["time"])),
            "std": float(np.std(eval_info[task_key]["time"]))
        }
    """
    eval_info["solved_tasks"] = max_tasks
    
                    
    eval_info_path = os.path.join(project_dir, "eval_info.yaml")
    with open(eval_info_path, "w") as eval_info_file:
        yaml.dump(eval_info, eval_info_file)

if __name__ == "__main__":
    
    
    
    parser = argparse.ArgumentParser(description="Postprocess evaluation results.")
    parser.add_argument("--project_dir", type=str, help="Path to the project directory.", default="projects/benchmarking/2025_07_06/stepping_gates_n_parity_n_input_2_episode_type_one-step_curriculum_False/brax_ppo/num_timesteps_5000000/MLP_num_layers_6_num_neurons_4")
    args = parser.parse_args()

    project_dir = args.project_dir    
    
    process_trials(project_dir)