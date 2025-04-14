
import jax
import equinox as eqx
import jax.numpy as jnp
from scripts.train_examples.tensorneat.tensorneat_utils import TensorneatExperiment
import other_frameworks.tensorneat
import numpy as onp
from source.other_frameworks.tensorneat.algorithm import NEAT
from source.other_frameworks.tensorneat.genome import DefaultGenome
from source.other_frameworks.tensorneat.genome.gene import DefaultNode, DefaultConn

from source.other_frameworks.tensorneat.genome.operations.mutation import DefaultMutation

from source.other_frameworks.tensorneat.genome.gene import DefaultConn, DefaultNode

from source.other_frameworks.tensorneat.common import ACT, AGG
import wandb
import matplotlib.pyplot as plt
from collections import defaultdict
from source.NDP_framework.base.utils.viz_utils import viz_heatmap, viz_policy_network
import networkx as nx
import pickle




class NeatExperiment(TensorneatExperiment):

    def __init__(self, env_config, model_config, exp_config, optimizer_config):
        super().__init__(env_config, model_config, exp_config, optimizer_config)


    def init_model(self):
        genome = DefaultGenome(
            num_inputs=self.config["env_config"]["observation_size"],
            num_outputs=self.config["env_config"]["action_size"],
            max_nodes=self.config["model_config"]["model_params"]["max_nodes"],
            output_transform=ACT.tanh,
            max_conns=self.config["model_config"]["model_params"]["max_nodes"]*4,
            #output_transform=ACT.sigmoid,

            mutation=DefaultMutation(conn_add=self.config["model_config"]["model_params"]["conn_add"],
                                     conn_delete=self.config["model_config"]["model_params"]["conn_delete"],
                                     node_add=self.config["model_config"]["model_params"]["node_add"],
                                     node_delete=self.config["model_config"]["model_params"]["node_delete"],
                                     ),

            node_gene=DefaultNode(
                activation_options=(ACT.tanh,),
                activation_default=ACT.tanh,
                #activation_options=(ACT.relu,),
                #activation_default=ACT.relu,
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
        plt.clf()
        return adj_matrix



    def viz_policies(self):
        fig, axes = plt.subplots(2, 5, figsize=(15, 8))
        axes = axes.flatten()

        pop_conns = self.final_state["state"].state_dict["pop_conns"]
        pop_nodes = self.final_state["state"].state_dict["pop_nodes"]

        fitnesses = onp.array(self.final_state["state"].state_dict["fitnesses"])
        species_ids = onp.array(
            self.final_state["state"].state_dict["species"].state_dict["idx2species"])

        species = defaultdict(list)
        for indiv_idx, indiv_conns in enumerate(onp.array(pop_conns)):
            species[species_ids[indiv_idx]].append( fitnesses[indiv_idx])
        species_idx = 0

        for _, indivs in species.items():
            # find indiv with highest fitness
            most_fit= onp.argmax(indivs)
            nodes = pop_nodes[most_fit, ...]
            conns = pop_conns[most_fit, ...]

            network = self.model.genome.network_dict(self.final_state["state"], nodes, conns)
            graph = self.model.genome.visualize(network,
                                        save_path="temp.png",
                                                ax=axes[int(species_idx)])
            """
            weights = nx.to_numpy_array(graph, weight="weight")
            
            viz_policy_network(weights=weights,
                               n_input=self.config["env_config"]["observation_size"],
                               n_output=self.config["env_config"]["action_size"],
                               filename=self.config["exp_config"]["trial_dir"] + "/visuals/population/networks/",
                               network_type=self.config["model_config"]["network_type"],
                               ax=axes[int(species_idx)])
            """
            axes[int(species_idx)].set_title("Fitness: " + str(indivs[most_fit]))

            species_idx +=1

            if species_idx > 9:
                break


        plt.tight_layout()
        filename = self.config["exp_config"]["trial_dir"] + "/visuals/population/policy_network.png"
        plt.savefig(filename, dpi=300)

        self.logger_run.track(filename, name='policy_network', format='png')  # Specify format like 'gif' or 'mp4'

        #wandb.log({"Best policy per species": fig})
        plt.clf()
        plt.close()

    def viz_final_policy(self):
        network = self.model.genome.network_dict(self.final_state["state"], *self.final_state["params"])
        graph = self.model.genome.visualize(network, save_path=self.config["exp_config"]["trial_dir"] + "/visuals/policy/network.png")
        plt.clf()
        super().viz_final_policy()

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



