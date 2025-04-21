""" Process all trials of a project
"""
import argparse
import yaml
from collections import defaultdict
import numpy as onp

from aim import Run, Repo

def get_num_gens(value, logger_hash):
        """ Loads wandb info to see when wat the first timestep where the maximum fitness was achieved
        """
        value = value + 1

        aim_dir = "."


        repo = Repo(aim_dir)
        query = "metric.name == 'current_task'" 

        # Get collection of metrics
        for run_metrics_collection in repo.query_metrics(query).iter_runs():
            if run_metrics_collection.run.hash == logger_hash:

                for metric in run_metrics_collection:
                    # Get run params
                    params = metric.run[...]
                    # Get metric values
                    steps, metric_values = metric.values.sparse_numpy()

        query = "metric.name == 'generation'"  # Example query

        # Get collection of metrics
        for run_metrics_collection in repo.query_metrics(query).iter_runs():
            if run_metrics_collection.run.hash == logger_hash:

                for metric in run_metrics_collection:
                    # Get run params
                    params = metric.run[...]
                    # Get metric values
                    steps, generations = metric.values.sparse_numpy()
        try:

            indices = onp.where(metric_values == value)

            if indices[0].size > 0:
                first_step = indices[0][0]  # Get the first row index
                return int(generations[first_step])

            else:
                print(f"The value {value} was not found in the array.")
                return None

        except UnboundLocalError:
            return None




def process_trial(project, trial, num_tasks):
    data_dir = project + "/trial_" + str(trial) + "/data/eval/info.yml"
    with open(data_dir, "r") as f:
        eval_info = yaml.safe_load(f)
        
    trial_stats = {}
        
    for key in eval_info.keys():
        rewards_mean = float(onp.mean(eval_info[key]["rewards"]))
        rewards_var = float(onp.var(eval_info[key]["rewards"]))
        
        success_mean = float(onp.mean(eval_info[key]["success"]))
        success_var = float(onp.var(eval_info[key]["success"]))

        trial_stats[key] = {"rewards_mean": rewards_mean,
                            "rewards_var": rewards_var,
                            "success_mean": success_mean,
                            "success_var": success_var,
                            } 
        
        
    # get information from aim
    aim_info = project + "/aim_hashes.yaml"
    with open(aim_info, "r") as f:
        aim_hashes = yaml.safe_load(f)
    trial_aim_hash = aim_hashes[trial]
    
    trial_stats["steps"] = {}
    
    for task in range(num_tasks):
        steps = get_num_gens(task, trial_aim_hash)
        
        if steps is not None:
        
            trial_stats["steps"]["task_" + str(task)] = steps
        
    return trial_stats       

    

def post_process(project):
    
    with open(project + "/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    
    total_stats = defaultdict(lambda: defaultdict(list))
    
    for trial in range(config["exp_config"]["num_trials"]):
        trial_stats = process_trial(project, trial, num_tasks=config["env_config"]["num_tasks"])
        
        for task in trial_stats.keys():
        
            for metric, value in trial_stats[task].items():
                if "var" not in metric:
                    total_stats[task][metric].append(value)   
            
        with open(project + "/trial_" + str(trial) + "/eval_info.yaml", "w") as f:
            yaml.dump(trial_stats, f)
            
    average_stats = {}
    for task in total_stats.keys():
        average_stats[task] = {}
        
        for metric, items in total_stats[task].items():
            average_stats[task][metric + "_mean"] = float(onp.mean(items))
            average_stats[task][metric + "_var"] = float(onp.var(items))

            
    with open(project + "/eval_info.yaml", "w") as f:
        yaml.dump(average_stats, f)
            
            
        
        
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script processes the data produced for a project.")
    parser.add_argument("--project", type=str, help="Project path", default="")
    args = parser.parse_args()
    
    
    project = args.project
    project = "/home/eleni/workspace/neuroevolution_in_transfer_learning/projects/benchmarking/2025_04_20/stepping_gates_n_parity_only_n_n_input_6_episode_type_one-step_curriculum_False/brax_ppo/num_timesteps_150000000/feedforward_num_layers_6_num_neurons_4"
    post_process(project)
    