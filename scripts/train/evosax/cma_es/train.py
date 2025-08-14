""" Script for training CMA-ES """
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
sys.path.insert(0, "methods")
sys.path.insert(0, "methods/evosax_wrapper") # to be able to import evosax
sys.path.insert(0, "scripts")
from scripts.train.evosax.train_utils import EvosaxExperiment as Experiment
import os
import envs
from scripts.train.base.utils import default_env_params
from scripts.train.evosax.cma_es.hyperparams import train_gens, hyperparams
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
    num_timesteps = train_gens[env_name]
    optimizer_config = {"optimizer_name": "cma_es",
                        "optimizer_type": "evosax",
                        "optimizer_params": {"generations": num_timesteps,
                                             "strategy": "CMA_ES",
                                             "popsize": 256}}
    
    model_config = {"network_type": "MLP",
                    "model_params": hyperparams[env_name]}


    exp = Experiment(env_config=env_config,
                     optimizer_config=optimizer_config,
                     model_config = model_config,
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
    num_timesteps = train_gens[env_name]
    optimizer_config = {"optimizer_name": "cma_es",
                        "optimizer_type": "evosax",
                        "optimizer_params": {"generations": num_timesteps,
                                             "strategy": "CMA_ES",
                                             "popsize": 256}}
    
    model_config = {"network_type": "MLP",
                    "model_params": hyperparams[env_name]}


    exp = Experiment(env_config=env_config,
                     optimizer_config=optimizer_config,
                     model_config = model_config,
                     exp_config=exp_config)
    exp.run()
    
def train_gymnax(num_trials, env_name):

    # configure experiment
    exp_config = {"seed": 0,
                  "num_trials": num_trials}
    
    # configure environment
    env_params = default_env_params[env_name]
    env_params["noise_range"] = 0.0
    env_config = {"env_type": "gymnax",
                  "env_name": env_name,
                  "curriculum": False,
                  "env_params": env_params}
    
    
    # configure method
    num_timesteps = train_gens[env_name]
    optimizer_name = "OpenES"
    optimizer_config = {"optimizer_name": optimizer_name,
                        "optimizer_type": "evosax",
                        "optimizer_params": {"generations": num_timesteps,
                                             "strategy": optimizer_name,
                                             "popsize": 1024}}
    
    model_config = {"network_type": "MLP",
                    "model_params": hyperparams[env_name]}


    exp = Experiment(env_config=env_config,
                     optimizer_config=optimizer_config,
                     model_config = model_config,
                     exp_config=exp_config)
    exp.run()

def train_stepping_gates_all(num_trials):
    train_stepping_gates(num_trials=num_trials, env_name="n_parity", curriculum=False)
    train_stepping_gates(num_trials=num_trials, env_name="n_parity_only_n", curriculum=True)
    train_stepping_gates(num_trials=num_trials, env_name="simple_alu", curriculum=True)



def train_ecorobot_all(num_trials):
    train_ecorobot(num_trials=num_trials, env_name="locomotion", robot_type="halfcheetah")
    train_ecorobot(num_trials=num_trials, env_name="locomotion", robot_type="ant")
    train_ecorobot(num_trials=num_trials, env_name="maze_with_stepping_stones", robot_type="discrete_fish")

    train_ecorobot(num_trials=num_trials, env_name="locomotion_with_obstacles", robot_type="halfcheetah")
    train_ecorobot(num_trials=num_trials, env_name="deceptive_maze_easy", robot_type="discrete_fish")
    train_ecorobot(num_trials=num_trials, env_name="deceptive_maze_easy", robot_type="ant")

def train_gymnax_all(num_trials):
    #train_gymnax(num_trials=num_trials, env_name="CartPole-v1")
    train_gymnax(num_trials=num_trials, env_name="MountainCar-v0")
    #train_gymnax(num_trials=num_trials, env_name="CartPole-v1")
    #train_gymnax(num_trials=num_trials, env_name="MountainCarContinuous-v0")







    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script trains Proximal Policy Optimisation on the stepping gates and ecorobot benchmarks")
    parser.add_argument("--num_trials", type=int, help="Number of trials", default=5)
    args = parser.parse_args()

    #train_stepping_gates_all(num_trials=args.num_trials)
    #train_ecorobot_all(num_trials=args.num_trials)
    train_gymnax_all(num_trials=args.num_trials)
