""" Script for training Neuroevolution of Augmenting Topologies"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
sys.path.insert(0, "methods")
sys.path.insert(0, "scripts")
import envs
from scripts.train.tensorneat.neat.train_utils import NEATExperiment as Experiment
from scripts.train.base.utils import default_env_params
from scripts.train.tensorneat.neat.hyperparams import train_gens, hyperparams
import argparse



def train_stepping_gates(num_trials, env_name, curriculum):
    
    
    # configure experiment
    exp_config = {"seed": 0,
                  "num_trials": num_trials}
    
    
        # configure environment


    env_params = default_env_params[env_name]
    env_params["episode_type"] = "full"
    env_params["curriculum"] = curriculum

    env_config = {"env_type": "stepping_gates",
                  "env_name": env_name,
                  "curriculum": curriculum,
                  "env_params": env_params}

    # configure method
    model_config = {"network_type": "MLP_with_skip",
                    "model_params": {"max_nodes": 100,
                                     **hyperparams[env_name]}}

    optimizer_config = {"optimizer_name": "neat",
                        "optimizer_type": "tensorneat",
                        "optimizer_params": {"generations": train_gens[env_name],
                                             "pop_size": 1024,
                                             "num_species": 20}}


    exp = Experiment(env_config=env_config,
                     model_config=model_config,
                     optimizer_config=optimizer_config,
                     exp_config=exp_config)
    exp.run()
    
    
def train_ecorobot(num_trials, env_name, robot_type):
    # configure experiment
    exp_config = {"seed": 0,
                  "num_trials": num_trials}
    
    
        # configure environment


    env_params = default_env_params[env_name]
    env_params["episode_type"] = "full"
    env_params["curriculum"] = False

    env_config = {"env_type": "ecorobot",
                  "env_name": env_name,
                  "curriculum": False,
                  "env_params": {"robot_type": robot_type}}

    # configure method
    model_config = {"network_type": "MLP_with_skip",
                    "model_params": {"max_nodes": 100,
                                     **hyperparams[env_name]}}

    optimizer_config = {"optimizer_name": "neat",
                        "optimizer_type": "tensorneat",
                        "optimizer_params": {"generations": train_gens[env_name],
                                             "pop_size": 1024,
                                             "num_species": 50}}


    exp = Experiment(env_config=env_config,
                     model_config=model_config,
                     optimizer_config=optimizer_config,
                     exp_config=exp_config)
    exp.run()
    


def train_ecorobot_all(num_trials):
    #train_ecorobot(num_trials=num_trials, env_name="locomotion", robot_type="halfcheetah")
    #train_ecorobot(num_trials=num_trials, env_name="locomotion", robot_type="ant")
    #train_ecorobot(num_trials=num_trials, env_name="maze_with_stepping_stones", robot_type="discrete_fish")

    train_ecorobot(num_trials=num_trials, env_name="locomotion_with_obstacles", robot_type="halfcheetah")
    train_ecorobot(num_trials=num_trials, env_name="deceptive_maze_easy", robot_type="discrete_fish")
    train_ecorobot(num_trials=num_trials, env_name="deceptive_maze_easy", robot_type="ant")



def train_stepping_gates_all(num_trials):
    train_stepping_gates(num_trials=num_trials, env_name="n_parity", curriculum=False)
    train_stepping_gates(num_trials=num_trials, env_name="n_parity_only_n", curriculum=True)
    train_stepping_gates(num_trials=num_trials, env_name="simple_alu", curriculum=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script trains Proximal Policy Optimisation on the stepping gates and ecorobot benchmarks")
    parser.add_argument("--num_trials", type=int, help="Number of trials", default=2)
    args = parser.parse_args()

    #train_stepping_gates_all(num_trials=args.num_trials)
    train_ecorobot_all(num_trials=args.num_trials)
