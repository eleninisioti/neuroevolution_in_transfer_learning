train_gens = {"n_parity": 5000,
              "n_parity_only_n": 5000,
                   "simple_alu": 10000,
                   "maze_with_stepping_stones": 1000,
                   "locomotion_with_obstacles": 1000,
                   "locomotion": 2000,
                   "deceptive_maze_easy": 100}



hyperparams = {
    "n_parity": { "max_hidden_neurons": 24,
                     "discrete_weights": False,
                 },
    "n_parity_only_n": { "max_hidden_neurons": 24,
                     "discrete_weights": False,
                 },
   "simple_alu": { "max_hidden_neurons": 24,
                     "discrete_weights": False,
                 },
    "maze_with_stepping_stones": { "max_hidden_neurons": 24,
                     "discrete_weights": False,
                 },
    "locomotion_with_obstacles": { "max_hidden_neurons": 24,
                     "discrete_weights": False,
                 },
    "deceptive_maze_easy": { "max_hidden_neurons": 24,
                     "discrete_weights": False,
                 },
}


