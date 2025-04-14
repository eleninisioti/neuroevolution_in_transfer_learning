
import jax
import equinox as eqx
import jax.numpy as jnp
from scripts.train_examples.tensorneat.tensorneat_utils import TensorneatExperiment
import other_frameworks.tensorneat
import numpy as onp
from source.other_frameworks.tensorneat.algorithm import NEAT
from source.other_frameworks.tensorneat.algorithm.hyperneat import HyperNEATFeedForward, HyperNEAT
from source.other_frameworks.tensorneat.algorithm.hyperneat.substrate import MLPSubstrate, FullSubstrate
from source.other_frameworks.tensorneat.genome.operations.mutation import DefaultMutation

from source.other_frameworks.tensorneat.genome import DefaultGenome
from source.other_frameworks.tensorneat.common import ACT, AGG
from scripts.train_examples.tensorneat.hyperneat.substrates import get_substrate
from NDP_framework.base.utils.viz_utils import viz_heatmap,  viz_policy_network, viz_histogram
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import wandb
from source.other_frameworks.tensorneat.genome.gene import DefaultConn, DefaultNode
import pickle


class HyperNeatExperiment(TensorneatExperiment):

    def __init__(self, env_config, model_config, exp_config, optimizer_config):
        super().__init__(env_config, model_config, exp_config, optimizer_config)



    def load_model(self, params):
        pop = jax.tree_map(lambda x:jnp.expand_dims(x, 0), params)


        pop_transformed = jax.vmap(self.final_state["algorithm"].transform, in_axes=(None,0))(
            self.final_state["state"], pop)  # this returns some info about nodes and connections that is useful for forward
        return jax.tree_map(lambda x : x[0,...], pop_transformed)

    def eval_task(self, policy_params, tasks, final_policy=False):
        pop_transformed = self.load_model(policy_params)
        state = self.final_state["state"]

        # Fix `y` and `z`
        # act_fn = partial(self.model.forward, state=state, transformed=pop_transformed)

        def act_fn(obs, action_size=None, obs_size=None):
            return self.model.forward(state=state, transformed=pop_transformed, inputs=obs)

        super().run_eval(jax.jit(act_fn), tasks, final_policy)



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



        if self.config["model_config"]["model_name"]=="RNN":

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
            species[species_ids[indiv_idx]].append(fitnesses[indiv_idx])
        species_idx = 0

        for _, indivs in species.items():
            # find indiv with highest fitness
            most_fit = onp.argmax(indivs)
            nodes = pop_nodes[most_fit, ...]
            conns = pop_conns[most_fit, ...]

            network = self.model.hyper_genome.network_dict(self.final_state["state"], nodes, conns)
            graph = self.model.hyper_genome.visualize(network,
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

            species_idx += 1

            if species_idx > 10:
                break

        plt.tight_layout()
        plt.savefig(self.config["exp_config"]["trial_dir"] + "/visuals/population/policy_networks.png", dpi=300)

        #wandb.log({"Best policy per species": fig})
        plt.clf()
        plt.close()


    def viz_final_policy(self):

        cppn_genome = self.model.neat.genome
        state = self.final_state["state"]
        best = self.final_state["params"]
        cppn_network = cppn_genome.network_dict(state, *best)
        graph = cppn_genome.visualize(cppn_network,
                                      save_path=self.config["exp_config"]["trial_dir"] + "/visuals/policy/cppn_tensorneat.png")
        plt.clf()
        cppn_matrix = nx.to_numpy_array(graph, weight="weight")

        # visualize hyperneat genome
        hyperneat_genome = self.model.hyper_genome
        # use cppn to calculate the weights in hyperneat genome
        # return seqs, nodes, conns, u_conns
        _, hyperneat_nodes, hyperneat_conns, _ = self.model.transform(state, best)
        # mutate the connection with weight 0 (to visualize the network rather the substrate)
        hyperneat_conns = jnp.where(
            hyperneat_conns[:, 2][:, None] == 0, jnp.nan, hyperneat_conns
        )
        hyperneat_network = hyperneat_genome.network_dict(
            state, hyperneat_nodes, hyperneat_conns
        )
        graph = hyperneat_genome.visualize(
            hyperneat_network, save_path=self.config["exp_config"]["trial_dir"] + "/visuals/policy/network.png"
        )



        super().viz_final_policy()

        viz_heatmap(cppn_matrix,
                    filename=self.config["exp_config"]["trial_dir"] + "/visuals/policy/cppn_heatmap")

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
                    params = pickle.load(f)
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



