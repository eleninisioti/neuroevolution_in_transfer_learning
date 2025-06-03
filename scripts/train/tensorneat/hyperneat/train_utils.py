

import jax
import equinox as eqx
import jax.numpy as jnp
from scripts.train.tensorneat.train_utils import TensorneatExperiment
import methods.tensorneat
import numpy as onp
from methods.tensorneat.algorithm import NEAT
from methods.tensorneat.algorithm.hyperneat import HyperNEATFeedForward, HyperNEAT
from methods.tensorneat.algorithm.hyperneat.substrate import MLPSubstrate, FullSubstrate
from methods.tensorneat.genome.operations.mutation import DefaultMutation
from methods.tensorneat.genome.gene import DefaultConn, DefaultNode

from methods.tensorneat.genome import DefaultGenome
from methods.tensorneat.common import ACT, AGG
from scripts.train.tensorneat.hyperneat.substrates import get_substrate

from methods.tensorneat.common import ACT, AGG
import matplotlib.pyplot as plt
from collections import defaultdict
import networkx as nx
import pickle
from scripts.train.base.visuals import viz_histogram, viz_heatmap


class HyperNEATExperiment(TensorneatExperiment):

    def __init__(self, env_config, model_config, exp_config, optimizer_config):
        super().__init__(env_config, model_config, exp_config, optimizer_config)



    def load_model(self, params):
        pop = jax.tree_map(lambda x:jnp.expand_dims(x, 0), params)


        pop_transformed = jax.vmap(self.final_state["algorithm"].transform, in_axes=(None,0))(
            self.final_state["state"], pop)  # this returns some info about nodes and connections that is useful for forward
        return jax.tree_map(lambda x : x[0,...], pop_transformed)



    def load_cppn(self):

        neat = self.final_state["algorithm"].neat
        transformed,_ = neat.transform(self.final_state["best"])
        conns = transformed[2][0,...]
        activations = transformed[1][...,-1]
        return (activations, conns)



    def viz_substrate(self):
        pass

    def init_model(self):

        """
        input_coords, hidden_coords, output_coords = get_substrate(env_name=self.config["env_config"]["env_name"],
                                  n_input=self.config["env_config"]["observation_size"],
                                  n_output=self.config["env_config"]["action_size"],
                                  n_hidden=self.config["model_config"]["model_params"]["hidden_size"])
        """



        if self.config["model_config"]["network_type"]=="RNN":

            input_coors, hidden_coors, output_coors = get_substrate(self.config["env_config"]["env_name"],
                                                                    n_input=self.config["env_config"]["observation_size"],
                                                                    n_output=self.config["env_config"]["observation_size"],
                                                                    n_hidden=self.config["model_config"]["model_params"]["hidden_size"])
            substrate = FullSubstrate(
                input_coors=input_coors,
                hidden_coors=hidden_coors,
                output_coors=output_coors,
            )

            genome = DefaultGenome(
                num_inputs=4,  # size of query coors
                num_outputs=1,
                init_hidden_layers=(),
                max_nodes= self.config["model_config"]["model_params"]["cppn_max_nodes"],
                max_conns=self.config["model_config"]["model_params"]["cppn_max_conns"],
                output_transform=ACT.tanh,
                node_gene=DefaultNode(
                    bias_mutate_rate=0.1,
                    response_mutate_rate=0.1
                ),
                conn_gene=DefaultConn(weight_mutate_rate=0.1)
            )
            neat = NEAT(
                genome=genome,
                pop_size=self.config["optimizer_config"]["optimizer_params"]["pop_size"],
                species_size=self.config["optimizer_config"]["optimizer_params"]["num_species"],
                survival_threshold=0.01
            )

            self.model = HyperNEAT(
                substrate=substrate,
                neat=neat,
                activation=ACT.tanh,
                output_transform=ACT.tanh,
                activate_time=10

            )

        else:
            layers = [self.config["env_config"]["observation_size"] + 1]

            for layer in range(self.config["model_config"]["model_params"]["num_hidden_layers"]):
                layers.append(self.config["model_config"]["model_params"]["num_hidden_neurons_per_layer"])
            layers.append(self.config["env_config"]["action_size"])

            genome = DefaultGenome(
                num_inputs=4,  # size of query coors
                num_outputs=1,
                init_hidden_layers=(),
                output_transform=ACT.tanh,
                mutation=DefaultMutation(conn_add=self.config["model_config"]["model_params"]["conn_add"],
                                     conn_delete=self.config["model_config"]["model_params"]["conn_delete"],
                                     node_add=self.config["model_config"]["model_params"]["node_add"],
                                     node_delete=self.config["model_config"]["model_params"]["node_delete"],
                                     ),
                node_gene=DefaultNode(
                    bias_mutate_rate=self.config["model_config"]["model_params"]["mut_node"],
                    response_mutate_rate=self.config["model_config"]["model_params"]["mut_node"],
                ),
                conn_gene=DefaultConn(weight_mutate_rate=self.config["model_config"]["model_params"]["mut_conn"])
            )
            substrate = MLPSubstrate(layers=layers, coor_range=(-max(layers), max(layers), -max(layers), max(layers)))
            neat = NEAT(
                genome=genome,
                pop_size=self.config["optimizer_config"]["optimizer_params"]["pop_size"],
                species_size=self.config["optimizer_config"]["optimizer_params"]["num_species"],
                survival_threshold=0.01
            )


            self.model = HyperNEATFeedForward(
                substrate=substrate,
                neat=neat,
                activation=ACT.tanh,
                output_transform=ACT.tanh,

            )





    def params_to_weights(self, params):
        state = self.final_state["state"]
        cppn_genome = self.model.neat.genome

        cppn_network = cppn_genome.network_dict(state, *params)
        graph = cppn_genome.visualize(cppn_network, save_path="cppn_network.svg")
        plt.clf()

        cppn_matrix = nx.to_numpy_array(graph, weight="weight")

        # visualize hyperneat genome
        hyperneat_genome = self.model.hyper_genome
        # use cppn to calculate the weights in hyperneat genome
        # return seqs, nodes, conns, u_conns
        _, hyperneat_nodes, hyperneat_conns, _ = self.model.transform(state, params)
        # mutate the connection with weight 0 (to visualize the network rather the substrate)
        hyperneat_conns = jnp.where(
            hyperneat_conns[:, 2][:, None] == 0, jnp.nan, hyperneat_conns
        )
        hyperneat_network = hyperneat_genome.network_dict(
            state, hyperneat_nodes, hyperneat_conns
        )
        graph = hyperneat_genome.visualize(
            hyperneat_network, save_path="hyperneat_network.svg"
        )
        adj_matrix = nx.to_numpy_array(graph, weight="weight")
        plt.clf()
        return cppn_matrix, adj_matrix

    def save_training_info(self):

        final_cppn, final_weights = self.params_to_weights(self.final_state["params"])
        checkpoint_weights = []
        checkpoint_cppns = []

        for task in range(self.config["env_config"]["num_tasks"]):
            try:
                with open(self.config["exp_config"]["trial_dir"] + "/data/train/checkpoints/params_task_" + str(
                        task) + ".pkl", "rb") as f:
                    _, params = pickle.load(f)
                    cppn, weights = self.params_to_weights(params)


                    checkpoint_weights.append(weights)
                    checkpoint_cppns.append(cppn)
            except FileNotFoundError:
                continue



        # save final policy matrix
        self.training_info = {"policy_network": {"final": final_weights,
                                                 "checkpoints": checkpoint_weights},
                              "cppn": {"final": final_cppn,
                                       "checkpoints": checkpoint_cppns}}

        super().save_training_info()

        # save growth_information


    def viz_training_policy(self):
        viz_heatmap(self.training_info["policy_network"]["weights"],
                    filename=self.config["exp_config"]["trial_dir"] + "/visuals/policy/heatmap_cppn")
        viz_policy_network(weights=self.training_info["policy_network"]["weights"],
                           n_input=self.config["env_config"]["observation_size"],
                           n_output=self.config["env_config"]["action_size"],
                           filename=self.config["exp_config"]["trial_dir"] + "/visuals/policy/network_cppn",
                           network_type=self.config["model_config"]["network_type"])

        super().viz_training_policy()



