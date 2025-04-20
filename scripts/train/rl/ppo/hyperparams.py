
# ------ total training time -------
train_timesteps = {"n_parity_only_n": 150_000_000,
                   "simple_alu": 450_000_000}

# ------ PPO hyperparams for brax implementation -------
hyperparams = {
    "n_parity_only_n": {"reward_scaling": 1,
                 "unroll_length": 1,
                 "num_updates_per_batch": 4,
                 "discounting": 1.0,
                 "learning_rate": 3e-4,
                 "num_minibatches": 32,
                 "normalize_observations": False,
                 "num_envs": 4096,
                 "num_evals": 50,
                 "batch_size": 2048,
                 "entropy_cost": 1e-2,
                 "action_repeat": 1,
                 },
    "simple_alu": {"reward_scaling": 1,
                   "unroll_length": 1,
                   "num_updates_per_batch": 4,
                   "discounting": 1.0,
                   "learning_rate": 3e-4,
                   "num_minibatches": 32,
                   "normalize_observations": False,
                   "num_envs": 4096,
                   "num_evals": 50,
                   "batch_size": 2048,
                   "entropy_cost": 1e-2,
                   "action_repeat": 1,
                 },
}

# ------ neural network architecture for policy network. just MLPs ------
arch = {"n_parity_only_n": {"num_layers": 4,
                     "num_neurons": 6}}



