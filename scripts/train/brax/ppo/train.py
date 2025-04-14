import sys
sys.path.append(".")
from scripts.train_examples.brax.ppo.train_utils import PPOExperiment as Experiment
import os
import envs
from scripts.train_examples.utils import default_env_params
from scripts.train_examples.brax.ppo.hyperparams import default_num_timesteps



def train_gates_param(exp_config, env_name, curriculum):
    if curriculum:
        env_name = env_name + "_curric"



    env_config = {"env_type": "stepping_gates",
                  "env_name": env_name,
                  "curriculum": curriculum,
                  "env_params": default_env_params[env_name]}


    num_layers_values = [4, 6,8]
    num_neurons_values = [16, 32]
    for num_layers in num_layers_values:
        for num_neurons in num_neurons_values:



            model_config = {"model_name": "feedforward",
                            "network_type": "MLP_with_skip",
                            "model_params": {"num_layers": num_layers,
                                             "num_neurons": num_neurons}}

            optimizer_config = {"optimizer_name": "ppo",
                                "optimizer_type": "brax",
                                "optimizer_params": {"num_timesteps": num_timesteps}}

            exp = Experiment(env_config=env_config,
                             model_config=model_config,
                             optimizer_config=optimizer_config,
                             exp_config=exp_config)
            exp.train()



def train_brax( env_name):
    exp_config = {"seed": 0,
                  "n_trials": 1,
                  "upload_benchmark": True}


    num_layers = 4
    num_neurons = 32
    env_params = default_env_params[env_name]
    num_timesteps = default_num_timesteps[env_name]


    env_config = {"env_type": "brax",
                  "env_name": env_name,
                  "curriculum": False,
                  "env_params": env_params}

    model_config = {"network_type": "feedforward",
                    "model_params": {"num_layers": num_layers,
                                     "num_neurons": num_neurons}}

    optimizer_config = {"optimizer_name": "ppo",
                        "optimizer_type": "brax",
                        "optimizer_params": {"num_timesteps": num_timesteps}}


    exp = Experiment(env_config=env_config,
                     model_config=model_config,
                     optimizer_config=optimizer_config,
                     exp_config=exp_config)
    exp.train()


def train_ecorobot( env_name, robot_type):

    exp_config = {"seed": 0,
                  "n_trials": 1,
                  "upload_benchmark": True}
    num_layers =4
    num_neurons = 32
    env_params = default_env_params[env_name]
    num_timesteps = default_num_timesteps[env_name]
    env_params["robot_type"] = robot_type
    env_config = {"env_type": "ecorobot",
                  "env_name": env_name,
                  "curriculum": False,
                  "env_params": env_params,
                  }

    model_config = {"network_type": "feedforward",
                    "model_params": {"num_layers": num_layers,
                                     "num_neurons": num_neurons}}

    optimizer_config = {"optimizer_name": "ppo",
                        "optimizer_type": "brax",
                        "optimizer_params": {"num_timesteps": num_timesteps}}


    exp = Experiment(env_config=env_config,
                     model_config=model_config,
                     optimizer_config=optimizer_config,
                     exp_config=exp_config)
    exp.train()

def train_gates(exp_config, env_name, curriculum):

    num_layers = 6
    num_neurons = 4
    env_params = default_env_params[env_name]
    env_params["mode"] = "one-step"
    env_params["reward_for_solved"] = "imperfect"

    env_params["curriculum"] = curriculum


    num_timesteps = default_num_timesteps[env_name]


    env_config = {"env_type": "stepping_gates",
                  "env_name": env_name,
                  "curriculum": curriculum,
                  "env_params": env_params}

    model_config = {"network_type": "MLP_with_skip",
                    "model_params": {"num_layers": num_layers,
                                     "num_neurons": num_neurons}}

    optimizer_config = {"optimizer_name": "ppo",
                        "optimizer_type": "brax",
                        "optimizer_params": {"num_timesteps": num_timesteps}}


    exp = Experiment(env_config=env_config,
                     model_config=model_config,
                     optimizer_config=optimizer_config,
                     exp_config=exp_config)
    exp.train()


def train_gym( env_name):

    exp_config = {"seed": 0,
                  "n_trials": 1,
                  "upload_benchmark": True}

    num_layers = 3
    num_neurons = 32
    env_params = default_env_params[env_name]

    env_params["curriculum"] = False


    num_timesteps = default_num_timesteps[env_name]


    env_config = {"env_type": "gym",
                  "env_name": env_name,
                  "curriculum": False,
                  "env_params": env_params}

    model_config = {"network_type": "MLP_with_skip",
                    "model_params": {"num_layers": num_layers,
                                     "num_neurons": num_neurons}}

    optimizer_config = {"optimizer_name": "ppo",
                        "optimizer_type": "brax",
                        "optimizer_params": {"num_timesteps": num_timesteps}}


    exp = Experiment(env_config=env_config,
                     model_config=model_config,
                     optimizer_config=optimizer_config,
                     exp_config=exp_config)
    exp.train()

def train_gates_parametric(exp_config, env_name, curriculum):

    num_layers_values = [3,4,5,6]
    num_neurons_values = [4,8,16]

    for num_layers in num_layers_values:
        for num_neurons in num_neurons_values:
            env_params = default_env_params[env_name]
            env_params["reward_for_solved"] = "imperfect"
            env_params["mode"] = "one-step"
            env_params["curriculum"] = curriculum


            num_timesteps = default_num_timesteps[env_name]


            env_config = {"env_type": "digital_gates",
                          "env_name": env_name,
                          "curriculum": curriculum,
                          "env_params": env_params}

            model_config = {"network_type": "MLP_with_skip",
                            "model_params": {"num_layers": num_layers,
                                             "num_neurons": num_neurons}}

            optimizer_config = {"optimizer_name": "ppo",
                                "optimizer_type": "brax",
                                "optimizer_params": {"num_timesteps": num_timesteps}}


            exp = Experiment(env_config=env_config,
                             model_config=model_config,
                             optimizer_config=optimizer_config,
                             exp_config=exp_config)
            exp.train()

def train():
    #train_gym(env_name="CartPole-v1")
    #train_gym(env_name="Acrobot-v1")
    #train_ecorobot(env_name="maze_with_stepping_stones_small", robot_type="discrete_fish")

    #train_ecorobot(env_name="deceptive_maze_easy", robot_type="ant")
    #train_ecorobot(exp_config, env_name="deceptive_maze", robot_type="ant")

    #train_ecorobot( env_name="maze_with_stepping_stones_small_conditioned", robot_type="discrete_fish")

    #train_ecorobot( env_name="deceptive_maze_easy", robot_type="discrete_fish")
    #train_ecorobot( env_name="deceptive_maze_easy", robot_type="ant")


    #train_ecorobot(env_name="locomotion_with_obstacles", robot_type="halfcheetah")
    train_ecorobot( env_name="locomotion", robot_type="ant")
    #train_ecorobot( env_name="locomotion", robot_type="halfcheetah")
    #train_brax( "halfcheetah")




    #train_ecorobot( env_name="locomotion", robot_type="discrete_fish")
    #train_ecorobot( env_name="locomotion", robot_type="halfcheetah")
    #train_ecorobot( env_name="locomotion", robot_type="ant")

    #train_ecorobot( env_name="maze_with_stepping_stones_big", robot_type="ant")



    #train_ecorobot(exp_config, env_name="deceptive_maze", robot_type="discrete_fish")
    #train_ecorobot(exp_config, env_name="deceptive_maze", robot_type="ant")
    











if __name__ == "__main__":

    gpu = str(sys.argv[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


    train()
