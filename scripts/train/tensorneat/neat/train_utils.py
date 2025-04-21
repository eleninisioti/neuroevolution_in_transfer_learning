
import jax
import equinox as eqx
import jax.numpy as jnp
from scripts.train.tensorneat.train_utils import TensorneatExperiment
import methods.tensorneat
import numpy as onp
from methods.tensorneat.algorithm import NEAT
from methods.tensorneat.genome import DefaultGenome
from methods.tensorneat.genome.gene import DefaultNode, DefaultConn
from methods.tensorneat.genome.operations.mutation import DefaultMutation
from methods.tensorneat.genome.gene import DefaultConn, DefaultNode
from methods.tensorneat.common import ACT, AGG
import matplotlib.pyplot as plt
from collections import defaultdict
import networkx as nx
import pickle
from scripts.train.base.visuals import viz_histogram, viz_heatmap




class NEATExperiment(TensorneatExperiment):

    def __init__(self, env_config, model_config, exp_config, optimizer_config):
        super().__init__(env_config, model_config, exp_config, optimizer_config)


    def init_model(self):
        genome = DefaultGenome(
            num_inputs=self.config["env_config"]["observation_size"],
            num_outputs=self.config["env_config"]["action_size"],
            max_nodes=self.config["model_config"]["model_params"]["max_nodes"],
            output_transform=ACT.tanh,
            max_conns=self.config["model_config"]["model_params"]["max_nodes"]*4,
            mutation=DefaultMutation(conn_add=self.config["model_config"]["model_params"]["conn_add"],
                                     conn_delete=self.config["model_config"]["model_params"]["conn_delete"],
                                     node_add=self.config["model_config"]["model_params"]["node_add"],
                                     node_delete=self.config["model_config"]["model_params"]["node_delete"],
                                     ),
            node_gene=DefaultNode(
                activation_options=(ACT.tanh,),
                activation_default=ACT.tanh,
                bias_mutate_rate=self.config["model_config"]["model_params"]["mut_node"],
                response_mutate_rate=self.config["model_config"]["model_params"]["mut_node"],

            ),
            conn_gene=DefaultConn(weight_mutate_rate=self.config["model_config"]["model_params"]["mut_conn"])
        )

        self.model = NEAT(
            genome=genome,
            pop_size=self.config["optimizer_config"]["optimizer_params"]["pop_size"],
            species_size=self.config["optimizer_config"]["optimizer_params"]["num_species"],
            survival_threshold=0.01
        )



    def load_model(self, params):
        pop = jax.tree_map(lambda x:jnp.expand_dims(x, 0), params)


        pop_transformed = jax.vmap(self.final_state["algorithm"].transform, in_axes=(None,0))(
            self.final_state["state"], pop)  # this returns some info about nodes and connections that is useful for forward
        return jax.tree_map(lambda x : x[0,...], pop_transformed)


    def _map_to_adjacency_matrix(self, A):
        # Convert the JAX array to a NumPy array for easier manipulation
        A = onp.array(A)
        # Create a boolean mask where rows with all NaNs are True
        mask = onp.all(onp.isnan(A), axis=1)
        # Invert the mask to get rows that do not have all NaNs
        A = A[~mask]

        # Identify the unique nodes to determine the size of the adjacency matrix
        unique_nodes = onp.unique(A[:, :2])
        num_nodes = len(unique_nodes)

        # Create a mapping from node id to matrix index
        node_to_index = {node: i for i, node in enumerate(unique_nodes)}

        # Initialize the adjacency matrix
        adjacency_matrix = onp.zeros((num_nodes, num_nodes))

        # Populate the adjacency matrix
        for row in A:
            input_node, output_node, edge_exists, edge_value = row
            if edge_exists:
                i = node_to_index[input_node]
                j = node_to_index[output_node]
                adjacency_matrix[i, j] = edge_value

        return adjacency_matrix

    def params_to_weights(self, params):
        state = self.final_state["state"]


        # visualize hyperneat genome
        hyperneat_genome = self.model.genome
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
        return adj_matrix



    def save_training_info(self):

        #weights = self._map_to_adjacency_matrix(conns)

        network = self.model.genome.network_dict(self.final_state["state"],  *self.final_state["params"])
        graph = self.model.genome.visualize(network, save_path="policy_network.svg")
        plt.clf()
        adj_matrix = nx.to_numpy_array(graph, weight="weight")

        checkpoint_weights = []
        for task in range(self.config["env_config"]["num_tasks"]):
            try:
                with open(self.config["exp_config"]["trial_dir"] + "/data/train/checkpoints/params_task_" + str(
                        task) + ".pkl", "rb") as f:
                    params = pickle.load(f)
                    checkpoint_weights.append(self.params_to_weights(params))
            except FileNotFoundError:
                continue

        # save final policy matrix
        self.training_info = {"policy_network": {"final": adj_matrix,
                                                 "checkpoints": checkpoint_weights}}
        super().save_training_info()

        # save growth_information



