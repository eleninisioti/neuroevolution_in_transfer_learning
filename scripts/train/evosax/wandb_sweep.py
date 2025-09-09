""" Script for Weights & Biases hyperparameter sweeping over evolutionary strategies """
import sys
import os
sys.path.append(".")
sys.path.insert(0, "methods")
sys.path.insert(0, "methods/evosax_wrapper") # to be able to import evosax
sys.path.insert(0, "scripts")
from scripts.train.evosax.train_utils import EvosaxExperiment as Experiment
import os
import envs
from scripts.train.base.utils import default_env_params
from scripts.train.evosax.cma_es.hyperparams import train_gens, hyperparams
import wandb
import argparse


def train_with_wandb():
    """Training function that will be called by W&B sweep"""
    
    # Initialize W&B run
    with wandb.init() as run:
        try:
            # Get hyperparameters from W&B
            config = wandb.config
            
            # configure environment
            env_params = default_env_params[config.env_name]
            env_params["noise_range"] = 0.0
            env_config = {"env_type": "craftax",
                          "env_name": config.env_name,
                          "curriculum": False,
                          "env_params": env_params}
            
            # configure optimizer
            optimizer_config = {"optimizer_name": config.strategy,
                                "optimizer_type": "evosax",
                                "optimizer_params": {"generations": train_gens[config.env_name],
                                                     "strategy": config.strategy,
                                                     "popsize": config.popsize,
                                                     "es_kws": {}}}
            
            # Add strategy-specific hyperparameters to es_kws
            if config.strategy == "SNES":
                optimizer_config["optimizer_params"]["es_kws"]["sigma_init"] = config.sigma_init
                optimizer_config["optimizer_params"]["es_kws"]["temperature"] = config.temperature
            elif config.strategy == "CMA_ES":
                optimizer_config["optimizer_params"]["es_kws"]["sigma_init"] = config.sigma_init
                optimizer_config["optimizer_params"]["es_kws"]["elite_ratio"] = config.elite_ratio
            elif config.strategy == "SimpleGA":
                optimizer_config["optimizer_params"]["es_kws"]["elite_ratio"] = config.elite_ratio
                optimizer_config["optimizer_params"]["es_kws"]["sigma_init"] = config.sigma_init
            elif config.strategy == "SAMR_GA":
                optimizer_config["optimizer_params"]["es_kws"]["elite_ratio"] = config.elite_ratio
                optimizer_config["optimizer_params"]["es_kws"]["sigma_init"] = config.sigma_init
                optimizer_config["optimizer_params"]["es_kws"]["sigma_meta"] = config.sigma_meta
            elif config.strategy == "OpenES":
                # OpenES uses the full es_kws structure from config
                optimizer_config["optimizer_params"]["es_kws"] = config.es_kws
            
            # Model config
            model_config = {"network_type": "MLP",
                            "model_params": hyperparams[config.env_name]}
            
            # Experiment config
            exp_config = {"seed": config.seed, "num_trials": config.num_trials}
            
            # Create experiment
            exp = Experiment(env_config=env_config,
                             optimizer_config=optimizer_config,
                             model_config=model_config,
                             exp_config=exp_config)
            
            # Run experiment
            exp.run()
            
        except RuntimeError as e:
            if "cuSolver invalid value error" in str(e) or "gpusolverDnSsyevd_bufferSize" in str(e):
                print(f"CUDA solver error encountered: {e}")
                print("This is likely due to numerical instability in CMA-ES")
                # Log the error to W&B
                wandb.log({"error": "cuda_solver_error", "error_message": str(e)})
                # Mark run as failed but don't crash the sweep
                wandb.finish(exit_code=1)
            else:
                # Re-raise other runtime errors
                raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            # Log the error to W&B
            wandb.log({"error": "unexpected_error", "error_message": str(e)})
            # Mark run as failed but don't crash the sweep
            wandb.finish(exit_code=1)


def create_openes_sweep_config(env_name, num_trials=1):
    """Create W&B sweep configuration for OpenES"""
    
    sweep_config = {
        "method": "grid",
        "name": f"openes_sweep_{env_name}",
        "metric": {
            "name": "current_best_fitness",
            "goal": "maximize"
        },
        "parameters": {
            "env_name": {"value": env_name},
            "num_trials": {"value": num_trials},
            "seed": {"values": [0, 42, 123]},
            "strategy": {"value": "OpenES"},
            "popsize": {
                "values": [128, 256, 512, 1024]
            },
            "es_kws": {
                "values": [
                    {
                        "lrate_init": 0.001,
                        "lrate_decay": 1.0,
                        "lrate_limit": 0.0001,
                        "sigma_init": 0.01,
                        "sigma_decay": 0.999,
                        "sigma_limit": 0.001
                    },
                    {
                        "lrate_init": 0.01,
                        "lrate_decay": 1.0,
                        "lrate_limit": 0.001,
                        "sigma_init": 0.01,
                        "sigma_decay": 0.999,
                        "sigma_limit": 0.001
                    },
                    {
                        "lrate_init": 0.1,
                        "lrate_decay": 1.0,
                        "lrate_limit": 0.01,
                        "sigma_init": 0.01,
                        "sigma_decay": 0.999,
                        "sigma_limit": 0.001
                    },
                    {
                        "lrate_init": 0.01,
                        "lrate_decay": 0.99,
                        "lrate_limit": 0.001,
                        "sigma_init": 0.01,
                        "sigma_decay": 0.99,
                        "sigma_limit": 0.001
                    },
                    {
                        "lrate_init": 0.01,
                        "lrate_decay": 0.95,
                        "lrate_limit": 0.001,
                        "sigma_init": 0.01,
                        "sigma_decay": 0.95,
                        "sigma_limit": 0.001
                    }
                ]
            }
        }
    }
    
    return sweep_config


def create_cmaes_sweep_config(env_name, num_trials=1):
    """Create W&B sweep configuration for CMA-ES"""
    
    sweep_config = {
        "method": "grid",
        "name": f"cmaes_sweep_{env_name}",
        "metric": {
            "name": "current_best_fitness",
            "goal": "maximize"
        },
        "parameters": {
            "env_name": {"value": env_name},
            "num_trials": {"value": num_trials},
            "seed": {"values": [0, 42, 123]},
            "strategy": {"value": "CMA_ES"},
            "popsize": {
                "values": [128, 256]
            },
            "sigma_init": {
                "values": [0.1, 0.2, 0.5]
            },
            "elite_ratio": {
                "values": [0.3, 0.5, 0.7]
            }
        }
    }
    
    return sweep_config


def create_simplega_sweep_config(env_name, num_trials=1):
    """Create W&B sweep configuration for SimpleGA"""
    
    sweep_config = {
        "method": "grid",
        "name": f"simplega_grid_sweep_{env_name}",
        "metric": {
            "name": "deepest_level",
            "goal": "maximize"
        },
        "parameters": {
            "env_name": {"value": env_name},
            "num_trials": {"value": num_trials},
            "seed": {"values": [0, 42, 123]},
            "strategy": {"value": "SimpleGA"},
            "popsize": {
                "values": [256]
            },
            "elite_ratio": {
                "values": [0.1, 0.2, 0.3, 0.5, 0.7]
            },
            "sigma_init": {
                "values": [0.01, 0.1, 0.5]
            }
        }
    }
    
    return sweep_config


def create_snes_sweep_config(env_name, num_trials=1):
    """Create W&B sweep configuration for SNES using grid search"""
    
    sweep_config = {
        "method": "grid",
        "name": f"snes_grid_sweep_{env_name}",
        "metric": {
            "name": "current_best_fitness",
            "goal": "maximize"

        },

        "parameters": {
            "env_name": {"value": env_name},
            "num_trials": {"value": num_trials},
            "seed": {"values": [0, 42, 123]},
            "strategy": {"value": "SNES"},
            "popsize": {
                "values": [256]
            },
            "sigma_init": {
                "values": [0.01, 0.1, 0.5, 1.0]
            },
            "temperature": {
                "values": [1,10,20,50,100]
            }
        }
    }
    
    return sweep_config


def create_samrga_sweep_config(env_name, num_trials=1):
    """Create W&B sweep configuration for SAMR_GA using grid search"""
    
    sweep_config = {
        "method": "grid",
        "name": f"samrga_grid_sweep_{env_name}",
        "metric": {
            "name": "deepest_level",
            "goal": "maximize"
        },
        "parameters": {
            "env_name": {"value": env_name},
            "num_trials": {"value": num_trials},
            "seed": {"values": [0, 42, 123]},
            "strategy": {"value": "SAMR_GA"},
            "popsize": {
                "values": [ 256]
            },
            "elite_ratio": {
                "values": [0.1, 0.2, 0.3, 0.5]
            },
            "sigma_init": {
                "values": [0.01, 0.1, 0.5]
            },
            "sigma_meta": {
                "values": [1.0, 2.0, 5.0]
            }
        }
    }
    
    return sweep_config


def run_openes_sweep(env_name, num_trials=1, project_name="neuroevolution_sweep"):
    """Initialize and run W&B OpenES sweep"""
    
    # Create sweep configuration
    sweep_config = create_openes_sweep_config(env_name, num_trials)
    
    # Initialize W&B
    wandb.login()
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"Created OpenES sweep with ID: {sweep_id}")
    print(f"Sweep configuration: {sweep_config}")
    
    # Run the sweep
    wandb.agent(sweep_id, train_with_wandb, count=None)  # count=None runs until completion


def run_cmaes_sweep(env_name, num_trials=1, project_name="neuroevolution_sweep"):
    """Initialize and run W&B CMA-ES sweep"""
    
    # Create sweep configuration
    sweep_config = create_cmaes_sweep_config(env_name, num_trials)
    
    # Initialize W&B
    wandb.login()
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"Created CMA-ES sweep with ID: {sweep_id}")
    print(f"Sweep configuration: {sweep_config}")
    
    # Run the sweep
    wandb.agent(sweep_id, train_with_wandb, count=None)  # count=None runs until completion


def run_simplega_sweep(env_name, num_trials=1, project_name="neuroevolution_sweep"):
    """Initialize and run W&B SimpleGA sweep"""
    
    # Create sweep configuration
    sweep_config = create_simplega_sweep_config(env_name, num_trials)
    
    # Initialize W&B
    wandb.login()
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"Created SimpleGA sweep with ID: {sweep_id}")
    print(f"Sweep configuration: {sweep_config}")
    
    # Run the sweep
    wandb.agent(sweep_id, train_with_wandb, count=None)  # count=None runs until completion


def run_snes_sweep(env_name, num_trials=1, project_name="neuroevolution_sweep"):
    """Initialize and run W&B SNES sweep"""
    
    # Create sweep configuration
    sweep_config = create_snes_sweep_config(env_name, num_trials)
    
    # Initialize W&B
    wandb.login()
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"Created SNES sweep with ID: {sweep_id}")
    print(f"Sweep configuration: {sweep_config}")
    
    # Run the sweep
    wandb.agent(sweep_id, train_with_wandb, count=None)  # count=None runs until completion


def run_samrga_sweep(env_name, num_trials=1, project_name="neuroevolution_sweep"):
    """Initialize and run W&B SAMR_GA sweep"""
    
    # Create sweep configuration
    sweep_config = create_samrga_sweep_config(env_name, num_trials)
    
    # Initialize W&B
    wandb.login()
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"Created SAMR_GA sweep with ID: {sweep_id}")
    print(f"Sweep configuration: {sweep_config}")
    
    # Run the sweep
    wandb.agent(sweep_id, train_with_wandb, count=None)  # count=None runs until completion


def run_both_sweeps(env_name, num_trials=1, project_name="neuroevolution_sweep"):
    """Run both OpenES and CMA-ES sweeps sequentially"""
    
    print(f"Running both OpenES and CMA-ES sweeps for {env_name}")
    print("=" * 60)
    
    # Run OpenES sweep first
    print("\n1. Starting OpenES sweep...")
    run_openes_sweep(env_name, num_trials, project_name)
    
    print("\n" + "=" * 60)
    
    # Run CMA-ES sweep second
    print("\n2. Starting CMA-ES sweep...")
    run_cmaes_sweep(env_name, num_trials, project_name)
    
    print(f"\nBoth sweeps completed for {env_name}")


def run_all_sweeps(env_name, num_trials=1, project_name="neuroevolution_sweep"):
    """Run all five strategies (OpenES, CMA-ES, SimpleGA, SNES, SAMR_GA) sequentially"""
    
    print(f"Running all five evolutionary strategies for {env_name}")
    print("=" * 80)
    
    # Run OpenES sweep first
    print("\n1. Starting OpenES sweep...")
    run_openes_sweep(env_name, num_trials, project_name)
    
    print("\n" + "=" * 80)
    
    # Run CMA-ES sweep second
    print("\n2. Starting CMA-ES sweep...")
    run_cmaes_sweep(env_name, num_trials, project_name)
    
    print("\n" + "=" * 80)
    
    # Run SimpleGA sweep third
    print("\n3. Starting SimpleGA sweep...")
    run_simplega_sweep(env_name, num_trials, project_name)
    
    print("\n" + "=" * 80)
    
    # Run SNES sweep fourth
    print("\n4. Starting SNES sweep...")
    run_snes_sweep(env_name, num_trials, project_name)
    
    print("\n" + "=" * 80)
    
    # Run SAMR_GA sweep fifth
    print("\n5. Starting SAMR_GA sweep...")
    run_samrga_sweep(env_name, num_trials, project_name)
    
    print(f"\nAll five sweeps completed for {env_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="W&B hyperparameter sweep over evolutionary strategies")
    parser.add_argument("--env", type=str, help="Environment to sweep", default="Breakout-MinAtar")
    parser.add_argument("--num_trials", type=int, help="Number of trials per configuration", default=1)
    parser.add_argument("--project", type=str, help="W&B project name", default="neuroevolution_sweep")
    parser.add_argument("--strategy", type=str, choices=["openes", "cmaes", "simplega", "snes", "samrga", "both", "all"], help="Strategy to sweep", default="snes")
    args = parser.parse_args()

    print(f"Starting W&B {args.strategy.upper()} sweep for {args.env}")
    print(f"Project: {args.project}")
    #run_snes_sweep("Freeway-MinAtar", args.num_trials, args.project)
    #run_snes_sweep("Asterix-MinAtar", args.num_trials, args.project)
    #run_simplega_sweep("craftax", args.num_trials, args.project)
    run_samrga_sweep("craftax", args.num_trials, args.project)

    
    if args.strategy == "openes":
        run_openes_sweep(args.env, args.num_trials, args.project)
    elif args.strategy == "cmaes":
        run_cmaes_sweep(args.env, args.num_trials, args.project)
    elif args.strategy == "simplega":
        #run_simplega_sweep(args.env, args.num_trials, args.project)
        #run_simplega_sweep("Breakout-MinAtar", args.num_trials, args.project)
        run_simplega_sweep("Asterix-MinAtar", args.num_trials, args.project)
    elif args.strategy == "snes":
        #run_snes_sweep("Breakout-MinAtar", args.num_trials, args.project)
        run_snes_sweep("Asterix-MinAtar", args.num_trials, args.project)
        #run_snes_sweep(args.env, args.num_trials, args.project)
        #run_snes_sweep(args.env, args.num_trials, args.project)
    elif args.strategy == "samrga":
        run_samrga_sweep(args.env, args.num_trials, args.project)
    elif args.strategy == "both":
        run_both_sweeps(args.env, args.num_trials, args.project)
    elif args.strategy == "all":
        run_all_sweeps(args.env, args.num_trials, args.project) 