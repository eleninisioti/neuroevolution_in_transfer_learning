""" Script for training Proximal Policy Optimisation"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
sys.path.insert(0, "methods")
sys.path.insert(0, "scripts")

from scripts.train.rl.ppo.train_utils import PPOExperiment as Experiment
import os
import envs
from scripts.train.base.utils import default_env_params
from scripts.train.rl.ppo.hyperparams import train_timesteps, arch
import argparse



def train_stepping_gates(num_trials, env_name, curriculum):

    # configure experiment
    exp_config = {"seed": 0,
                  "num_trials": num_trials}
    
    # configure environment
    env_params = default_env_params[env_name]
    env_params["episode_type"] = "one-step"
    env_params["curriculum"] = curriculum
    env_config = {"env_type": "stepping_gates",
                  "env_name": env_name,
                  "curriculum": curriculum,
                  "env_params": env_params}
    
    
    # configure method
    num_timesteps = train_timesteps[env_name]
    optimizer_config = {"optimizer_name": "ppo",
                        "optimizer_type": "brax",
                        "optimizer_params": {"num_timesteps": num_timesteps}}
    
    model_config = {"network_type": "MLP",
                    "model_params": arch[env_name]}


    exp = Experiment(env_config=env_config,
                     optimizer_config=optimizer_config,
                     model_config = model_config,
                     exp_config=exp_config)
    exp.run()



def train_brax(num_trials, env_name):

    # configure experiment
    exp_config = {"seed": 0,
                  "num_trials": num_trials}
    
    # configure environment
    env_params = default_env_params[env_name]
    env_config = {"env_type": "brax",
                  "env_name": env_name,
                  "curriculum": False,
                  "env_params": env_params}
    
    
    # configure method
    num_timesteps = train_timesteps[env_name]
    optimizer_config = {"optimizer_name": "ppo",
                        "optimizer_type": "brax",
                        "optimizer_params": {"num_timesteps": num_timesteps}}
    
    model_config = {"network_type": "MLP",
                    "model_params": arch[env_name]}


    exp = Experiment(env_config=env_config,
                     optimizer_config=optimizer_config,
                     model_config = model_config,
                     exp_config=exp_config)
    exp.run()
    
    
def train_ecorobot(num_trials, env_name):

    # configure experiment
    exp_config = {"seed": 0,
                  "num_trials": num_trials}
    
    # configure environment
    env_params = default_env_params[env_name]
    env_config = {"env_type": "ecorobot",
                  "env_name": env_name,
                  "curriculum": False,
                  "env_params": env_params}
    
    
    # configure method
    num_timesteps = train_timesteps[env_name]
    optimizer_config = {"optimizer_name": "ppo",
                        "optimizer_type": "brax",
                        "optimizer_params": {"num_timesteps": num_timesteps}}
    
    model_config = {"network_type": "MLP",
                    "model_params": arch[env_name]}


    exp = Experiment(env_config=env_config,
                     optimizer_config=optimizer_config,
                     model_config = model_config,
                     exp_config=exp_config)
    exp.run()



def train_stepping_gates_all(num_trials):
    train_stepping_gates(num_trials=num_trials, env_name="n_parity_only_n", curriculum=False)
    train_stepping_gates(num_trials=num_trials, env_name="n_parity_only_n", curriculum=True)
    train_stepping_gates(num_trials=num_trials, env_name="simple_alu", curriculum=True)

def train_brax_all(num_trials):
    train_brax(num_trials=num_trials, env_name="halfcheetah")
    
    
    
def train_ecorobot_all(num_trials):
    train_ecorobot(num_trials=num_trials, env_name="locomotion", robot_type="halfcheetah")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script trains Proximal Policy Optimisation on the stepping gates and ecorobot benchmarks")
    parser.add_argument("--num_trials", type=int, help="Number of trials", default=10)
    args = parser.parse_args()

    #train_stepping_gates_all(num_trials=args.num_trials)
    train_brax_all(num_trials=args.num_trials)
    #train_ecorobot_all(num_trials=args.num_trials)
