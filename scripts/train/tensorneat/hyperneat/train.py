""" Script for training Neuroevolution of Augmenting Topologies"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
sys.path.insert(0, "methods")
sys.path.insert(0, "scripts")
import envs
from scripts.train.tensorneat.hyperneat.train_utils import HyperNEATExperiment as Experiment
from scripts.train.base.utils import default_env_params
from scripts.train.tensorneat.hyperneat.hyperparams import train_gens, hyperparams
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

    optimizer_config = {"optimizer_name": "hyperneat",
                        "optimizer_type": "tensorneat",
                        "optimizer_params": {"generations": train_gens[env_name],
                                             "pop_size": 1024,
                                             "num_species": 20}}


    exp = Experiment(env_config=env_config,
                     model_config=model_config,
                     optimizer_config=optimizer_config,
                     exp_config=exp_config)
    exp.run()





def train_all(num_trials):
    train_stepping_gates(num_trials=num_trials, env_name="n_parity_only_n", curriculum=False)
    train_stepping_gates(num_trials=num_trials, env_name="n_parity_only_n", curriculum=True)
    train_stepping_gates(num_trials=num_trials, env_name="simple_alu", curriculum=True)








if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script trains Proximal Policy Optimisation on the stepping gates and ecorobot benchmarks")
    parser.add_argument("--num_trials", type=int, help="Number of trials", default=10)
    args = parser.parse_args()

    train_all(num_trials=args.num_trials)
