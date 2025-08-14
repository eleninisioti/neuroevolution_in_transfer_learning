default_env_params = {"n_parity": {"n_input": 6},
                      "n_parity_only_n": {"n_input": 6},
                      "simple_alu": {},
                      "halfcheetah": {},
                      "locomotion": {},
                      "ant": {},    
                      "locomotion_with_obstacles": {},
                      "deceptive_maze_easy": {},
                      "maze_with_stepping_stones": {},
                      "hunted": {},
                      "Acrobot-v1": {"max_steps_in_episode": 200},
                      "MountainCar-v0": {"max_steps_in_episode": 200},
                      "MountainCarContinuous-v0": {"max_steps_in_episode": 200},
                      "CartPole-v1": {"max_steps_in_episode": 200},
                      }

max_rewards = {"n_parity": 0,
               "n_parity_only_n": 0,
               "simple_alu": 0,
               "halfcheetah": 4500,
               "ant": 4500,
               "locomotion": 4500,
               "locomotion_with_obstacles": 4500,
               "deceptive_maze_easy": 4500,
               "maze_with_stepping_stones": 4500,
               "hunted": 4500,
               "Acrobot-v1": 500,
               "MountainCar-v0": -50, # sota is -110
               "MountainCarContinuous-v0": -50,
               "CartPole-v1": 500,
               }


