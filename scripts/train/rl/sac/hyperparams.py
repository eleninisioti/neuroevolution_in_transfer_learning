

# how many timesteps to run PPO in each environment
default_num_timesteps = {
                         "n_parity_singletask": 150_000_000,
                         "stepping_gates": 450_000_000,
    "stepping_gates_all_tasks": 450_000_000,
"LunarLander-v2": 450_000_000,
    "foraging_with_sensors": 500_000,
                         "ant": 150_000_000,
                         "2d_nav_switching": 850_000_000,
                         "random_leg_ant": 150_000_000,
"locomotion_with_obstacles": 7_000_000,
"locomotion": 7_000_000,
"n_parity_all_tasks": 7_000_000,
"n_parity": 7_000_000,
"stepping_gates_all_tasks": 7_000_000,
"stepping_gates": 7_000_000,
"maze_with_stepping_stones_big": 7_000_000,
"maze_with_stepping_stones_small": 7_000_000,
"deceptive_maze": 7_000_000,
"deceptive_maze_easy": 7_000_000}





gen_hyperparams = {


    "locomotion": {
                   "num_evals": 20,
                   "reward_scaling": 30,
                   "normalize_observations": True,
                   "action_repeat": 1,
                   "discounting": 0.997,
                   "learning_rate": 6e-4,
                   "num_envs": 128,
                   "batch_size": 512,
                   "grad_updates_per_step": 64,
                   "max_devices_per_host": 1,
                   "max_replay_size": 1048576,
                   "min_replay_size": 8192},
    "n_parity_all_tasks": {
        "num_evals": 20,
        "reward_scaling": 30,
        "normalize_observations": True,
        "action_repeat": 1,
        "discounting": 0.997,
        "learning_rate": 6e-4,
        "num_envs": 128,
        "batch_size": 512,
        "grad_updates_per_step": 64,
        "max_devices_per_host": 1,
        "max_replay_size": 1048576,
        "min_replay_size": 8192},
    "stepping_gates_all_tasks": {
        "num_evals": 20,
        "reward_scaling": 30,
        "normalize_observations": True,
        "action_repeat": 1,
        "discounting": 0.997,
        "learning_rate": 6e-4,
        "num_envs": 128,
        "batch_size": 512,
        "grad_updates_per_step": 64,
        "max_devices_per_host": 1,
        "max_replay_size": 1048576,
        "min_replay_size": 8192},
    "stepping_gates": {
        "num_evals": 20,
        "reward_scaling": 30,
        "normalize_observations": True,
        "action_repeat": 1,
        "discounting": 0.997,
        "learning_rate": 6e-4,
        "num_envs": 128,
        "batch_size": 512,
        "grad_updates_per_step": 64,
        "max_devices_per_host": 1,
        "max_replay_size": 1048576,
        "min_replay_size": 8192}






}
