import jax
import jax.numpy as jnp
import numpy as onp
from source.NDP_framework.base.utils.viz_utils import  viz_policy_network
from envs.brax.brax.training.agents.sac import train as sac
import functools
from source.NDP_framework.base.utils.exp_utils import Experiment
import os
import envs
import pickle
from scripts.train_examples.brax.sac.hyperparams import gen_hyperparams
from functools import partial

def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)

class SACExperiment(Experiment):

    def __init__(self, env_config, model_config, exp_config, optimizer_config):
        super().__init__(env_config, model_config, exp_config, optimizer_config)




    def setup_keys(self):
        pass

    def setup_misc(self):

        hyperparams = gen_hyperparams[self.config["env_config"]["env_name"]]
        self.default_config = {"optimizer_config": {"optimizer_params": hyperparams},
                               "exp_config": {},
                               "model_config": {"model_params": {}},
                               "env_config": {"env_params": {}}}

    def init_model(self):

        if "episode_length" not in self.config["optimizer_config"]["optimizer_params"]:
            self.config["optimizer_config"]["optimizer_params"]["episode_length"] = self.config["env_config"]["episode_length"]

        train_fn = functools.partial(sac.train, **self.config["optimizer_config"]["optimizer_params"],
                                     **self.config["model_config"]["model_params"])
        self.model = functools.partial(train_fn.func, **train_fn.keywords)

    def cleanup(self):
        pass

    def save_params(self, training_state):

        def callback(training_state):
            env_params = training_state.env_params
            current_task = jnp.ravel(env_params).astype(jnp.int32)[0]
            with open(self.config["exp_config"]["trial_dir"] + "/data/train/checkpoints/params_task_" + str(current_task) + ".pkl",
                      "wb") as f:
                pickle.dump(        _unpmap(
            (training_state.normalizer_params, training_state.policy_params)), f)

        jax.debug.callback(callback, training_state)


    def progress(self, info):
        gen, env_params, metrics = info

        def log(wandb_info):
            logging_info = {
                "current_best_fitness": wandb_info["fitness"],
                "generation": wandb_info["gen"],
                "current_task": wandb_info["current_task"]
            }
            for key, value in logging_info.items():
                self.logger_run.track(value, name=key)

        total_eval_info = {
            "fitness": metrics["eval/episode_reward"],
            "gen": gen,
            "current_task": env_params[0][0]}
        log(total_eval_info)

    def eval_task(self, policy_params, tasks, final_policy=False):
        inference_fn = self.final_state["inference_fn"](policy_params)

        act_fn = partial(jax.jit(inference_fn), key_sample=jax.random.PRNGKey(0))
        super().run_eval(act_fn, tasks, final_policy)

    def get_final_policy(self):
        return self.training_info["policy_network"]["final"]
    def viz_final_policy(self):

        super().viz_final_policy()
        viz_policy_network(weights=self.training_info["policy_network"]["final"],
                           n_input=self.config["env_config"]["observation_size"],
                           n_output=self.config["env_config"]["action_size"],
                           filename=self.config["exp_config"]["trial_dir"] + "/visuals/policy/network_final",
                           network_type=self.config["model_config"]["network_type"])

        for task_idx, weights in enumerate(self.training_info["policy_network"]["checkpoints"]):
            viz_policy_network(weights=self.training_info["policy_network"]["final"],
                               n_input=self.config["env_config"]["observation_size"],
                               n_output=self.config["env_config"]["action_size"],
                               filename=self.config["exp_config"]["trial_dir"] + "/visuals/policy/network_" + str(task_idx),
                               network_type=self.config["model_config"]["network_type"])


    def run_trial(self):
        self.init_run()

        make_inference_fn, params, _, training_state = self.model(environment=self.env,
                                                                  progress_fn=self.progress,
                                                                  save_params_fn=self.save_params,
                                                                  seed=self.config["exp_config"]["trial_seed"])

        self.final_state = {"inference_fn": make_inference_fn,
                            "params": params,
                            "env_params": training_state.env_params,
                            }


    def params_to_weights(self, params):
        width = 0
        height = 0
        kernels = []
        for el in params[1]["params"].values():
            kernel = onp.array(el["kernel"])
            bias = onp.array(el["bias"])
            # kernel = onp.vstack((kernel, bias))
            height += kernel.shape[1]
            width += kernel.shape[0]
            kernels.append(kernel)
        num_neurons = width + self.env.action_size * 2  # because ppo outputs mean and sigma
        weights = jnp.zeros((num_neurons, num_neurons))
        start_x = 0
        start_y = self.config["env_config"]["observation_size"]
        for kernel in kernels:
            print(start_x, start_x + kernel.shape[0])
            print(start_y, start_y + kernel.shape[1])
            weights = weights.at[start_x:start_x + kernel.shape[0], start_y:start_y + kernel.shape[1]].set(kernel)
            start_x += kernel.shape[0]
            start_y += kernel.shape[1]

        # process bias
        width = 0
        for el in params[1]["params"].values():
            kernel = el["bias"]
            width += kernel.shape[0]
        width = width + self.env.observation_size
        bias = jnp.zeros((width,))
        start_x = self.env.observation_size
        for el in params[1]["params"].values():
            bias = bias.at[start_x:start_x + el["bias"].shape[0]].set(el["bias"])
            start_x += el["bias"].shape[0]

        bias = onp.array(bias)

        weights = onp.array(weights)
        weights = onp.vstack((bias[onp.newaxis, :], weights))
        weights = onp.hstack((onp.zeros((width + 1, 1)), weights))
        return weights

    def save_training_info(self):

        # weights = self._map_to_adjacency_matrix(conns)
        params = self.final_state["params"]
        weights = self.params_to_weights(params)

        checkpoint_weights = []
        for task in range(self.config["env_config"]["num_tasks"]):

            try:

                with open(self.config["exp_config"]["trial_dir"] + "/data/train/checkpoints/params_task_" + str(
                        task) + ".pkl","rb") as f:
                    params = pickle.load(f)
                    checkpoint_weights.append(self.params_to_weights(params))
            except FileNotFoundError:
                continue

        # save final policy matrix
        self.training_info = {"policy_network": {"final": weights,
                                                 "checkpoints": checkpoint_weights}}

        super().save_training_info()

