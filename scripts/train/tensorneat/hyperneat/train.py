import sys
sys.path.append(".")
from scripts.train_examples.tensorneat.hyperneat.train_utils import HyperNeatExperiment as Experiment
import os
import itertools
from scripts.train_examples.utils import default_env_params
from scripts.train_examples.tensorneat.hyperparams import default_num_gens






def train_gates_parametric(exp_config, curriculum=False):

    mut_node_values = [0.01, 0.1, 0.2]
    mut_conn_values = [0.01, 0.1, 0.2]
    node_add_values = [ 0.1]
    conn_add_values = [ 0.2]
    node_delete_values = [ 0.1]
    conn_delete_values = [ 0.2]
    num_hidden_layers = [1,2,4]
    num_hidden_neurons_per_layer = [8, 16, 32]


    all_combinations = list(
        itertools.product(mut_node_values, mut_conn_values, node_add_values, conn_add_values, node_delete_values, conn_delete_values, num_hidden_layers, num_hidden_neurons_per_layer))

    # Convert combinations into a list of dictionaries
    all_hyperparams = [
        {"mut_node": mut_node,
         "mut_conn": mut_conn,
         "node_add":  node_add,
         "conn_add": conn_add,
         "node_delete": node_add,
         "conn_delete": conn_add,
         "num_hidden_layers": num_hidden_layers,
         "num_hidden_neurons_per_layer": num_hidden_neurons_per_layer,

         }
        for mut_node, mut_conn, node_add, conn_add, node_delete, conn_delete, num_hidden_layers, num_hidden_neurons_per_layer in all_combinations]



    if curriculum:
        env_name  = "n_parity_curric"
    else:
        env_name  = "n_parity"
    for hyperparams in all_hyperparams:


        env_config = {"env_type": "digital_gates",
                      "env_name": env_name,
                      "curriculum": curriculum,
                      "env_params": {"n_input": 6}}

        model_config = {"model_name": "MLP_with_skip",

                        "model_params": {"cppn_max_nodes": 100,
                                         "cppn_max_conns": 1000,
                                         **hyperparams}}



        optimizer_config = {"optimizer_name": "hyperneat",
                            "optimizer_type": "tensorneat",
                            "optimizer_params": {"generations": 5000,
                                                 "pop_size": 5012,
                                                 "num_species": 50}}


    exp = Experiment(env_config=env_config,
                     model_config=model_config,
                     optimizer_config=optimizer_config,
                     exp_config=exp_config)
    exp.train()


def train_gymnax(env_name, curriculum=False):

    exp_config = {"seed": 0,
                  "n_trials": 2,
                  "upload_benchmark": True}

    env_params = default_env_params[env_name]

    hyperparams = {"num_hidden_neurons_per_layer": 16,
                   "num_hidden_layers": 1,
                   "mutation_node": 0.1,
                   "mutation_conn": 0.1
                   }


    env_config = {"env_type": "gymnax",
                  "env_name": env_name,
                  "curriculum": curriculum,
                  "env_params": env_params}

    model_config = {"network_type": "MLP_with_skip",

                    "model_params": {"cppn_max_nodes": 100,
                                     "cppn_max_conns": 1000,
                                     **hyperparams}}



    optimizer_config = {"optimizer_name": "hyperneat",
                        "optimizer_type": "tensorneat",
                        "optimizer_params": {"generations": default_num_gens[env_name],
                                             "pop_size": 5012,
                                             "num_species": 50}}


    exp = Experiment(env_config=env_config,
                     model_config=model_config,
                     optimizer_config=optimizer_config,
                     exp_config=exp_config)
    exp.train()


def train_ecorobot(env_name, robot_type, curriculum=False):

    exp_config = {"seed": 0,
                  "n_trials": 1,
                  "upload_benchmark": True}
    env_params = default_env_params[env_name]

    hyperparams = {"num_hidden_neurons_per_layer": 16,
                   "num_hidden_layers": 1,
                   "mutation_node": 0.1,
                   "mutation_conn": 0.1
                   }
    env_params["robot_type"] = robot_type


    env_config = {"env_type": "ecorobot",
                  "env_name": env_name,
                  "curriculum": curriculum,
                  "env_params": env_params}

    model_config = {"network_type": "MLP_with_skip",

                    "model_params": {"cppn_max_nodes": 100,
                                     "cppn_max_conns": 1000,
                                     **hyperparams}}



    optimizer_config = {"optimizer_name": "hyperneat",
                        "optimizer_type": "tensorneat",
                        "optimizer_params": {"generations": default_num_gens[env_name],
                                             "pop_size": 1024,
                                             "num_species": 20}}


    exp = Experiment(env_config=env_config,
                     model_config=model_config,
                     optimizer_config=optimizer_config,
                     exp_config=exp_config)
    exp.train()


def train_gates(exp_config,env_name, curriculum=False):

    exp_config = {"seed": 0,
                  "n_trials": 3,
                  "upload_benchmark": True}


    env_params = default_env_params[env_name]
    env_params["mode"] = "full"
    env_params["curriculum"] = curriculum


    hyperparams = {"num_hidden_neurons_per_layer": 16,
                   "num_hidden_layers": 1,
                   "mutation_node": 0.1,
                   "mutation_conn": 0.1
                   }


    env_config = {"env_type": "digital_gates",
                  "env_name": env_name,
                  "curriculum": curriculum,
                  "env_params": env_params}

    model_config = {"network_type": "MLP_with_skip",

                    "model_params": {"cppn_max_nodes": 100,
                                     "cppn_max_conns": 1000,
                                     **hyperparams}}



    optimizer_config = {"optimizer_name": "hyperneat",
                        "optimizer_type": "tensorneat",
                        "optimizer_params": {"generations": default_num_gens[env_name],
                                             "pop_size": 5012,
                                             "num_species": 50}}


    exp = Experiment(env_config=env_config,
                     model_config=model_config,
                     optimizer_config=optimizer_config,
                     exp_config=exp_config)
    exp.train()



def train_gates_curric(exp_config):

    hidden_size_values = [16]

    for hidden_size in hidden_size_values:


        env_config = {"env_type": "digital_gates",
                      "env_name": "n_parity_curric",
                      "env_params": {"n_input": 6}}

        model_config = {"model_name": "MLP_with_skip",
                        #"network_type": "MLP_with_skip",
                        #"network_type": "RNN",

                        "model_params": {"cppn_max_nodes": 100,
                                         "cppn_max_conns": 1000,
                                         "hidden_size": hidden_size}}

        optimizer_config = {"optimizer_name": "hyperneat",
                            "optimizer_type": "tensorneat",
                            "optimizer_params": {"generations": 5000,
                                                 "pop_size": 5012,
                                                 "num_species": 50}}


    exp = Experiment(env_config=env_config,
                     model_config=model_config,
                     optimizer_config=optimizer_config,
                     exp_config=exp_config)
    exp.train()

def train():
    #train_gymnax(env_name="CartPole-v1")
    #train_gymnax(env_name="Acrobot-v1")
    #quit()

    #train_ecorobot( env_name="deceptive_maze_easy", robot_type="discrete_fish")
    #train_ecorobot(env_name="deceptive_maze_easy", robot_type="ant")
    #train_ecorobot( env_name="maze_with_stepping_stones_small", robot_type="discrete_fish")

    train_ecorobot( env_name="locomotion_with_obstacles", robot_type="halfcheetah")



    #train_ecorobot( env_name="locomotion_with_obstacles", robot_type="halfcheetah")
    #train_ecorobot( env_name="locomotion_with_obstacles", robot_type="ant")

    #train_ecorobot( env_name="locomotion", robot_type="discrete_fish")
    #train_ecorobot(env_name="locomotion", robot_type="halfcheetah")
    #train_ecorobot( env_name="locomotion", robot_type="ant")

    #train_ecorobot( env_name="maze_with_stepping_stones_big", robot_type="ant")



    #train_ecorobot( env_name="deceptive_maze", robot_type="discrete_fish")
    #train_ecorobot( env_name="deceptive_maze", robot_type="ant")







if __name__ == "__main__":

    gpu = str(sys.argv[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu



    train()
