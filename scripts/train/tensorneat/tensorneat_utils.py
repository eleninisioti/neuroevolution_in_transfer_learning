import sys

import matplotlib.pyplot as plt

sys.path.append(".")
import envs
import source
from NDP_framework.base.utils.exp_utils import Experiment
import yaml
from brax import envs as brax_envs
from ecorobot import envs as ecorobot_envs
import jax
import equinox as eqx
import jax.numpy as jnp
import pickle
from brax.io import html
import os
import evosax
from source.other_frameworks.tensorneat.pipeline import Pipeline
from source.other_frameworks.tensorneat.problem.func_fit.n_parity import Nparity
from  source.other_frameworks.tensorneat.problem.rl.gates_env import GatesEnv
from  source.other_frameworks.tensorneat.problem.rl.ecorobot_env import EcorobotEnv
from  source.other_frameworks.tensorneat.problem.rl.gymnax_env import GymnaxEnv

import wandb
import numpy as onp
from NDP_framework.base.utils.viz_utils import viz_heatmap,  viz_policy_network, viz_histogram
from collections import defaultdict
import json
import pandas as pd
from functools import partial


class TensorneatExperiment(Experiment):

    def __init__(self, env_config, model_config, exp_config, optimizer_config):
        super().__init__(env_config, model_config, exp_config, optimizer_config)


    def setup_digital_gates_env(self):

        self.env = GatesEnv(self.config["env_config"])


        self.config["env_config"]["action_size"] = self.env.action_size
        self.config["env_config"]["observation_size"] = self.env.observation_size
        self.config["env_config"]["episode_length"] = self.env.episode_length


        return self.env


    def setup_ecorobot_env(self):

        self.env = EcorobotEnv(self.config["env_config"])
        self.config["env_config"]["action_size"] = self.env.action_size
        self.config["env_config"]["observation_size"] = self.env.observation_size
        self.config["env_config"]["episode_length"] = self.env.episode_length
        return self.env

    def setup_gymnax_env(self):

        self.env = GymnaxEnv(self.config["env_config"])
        self.config["env_config"]["action_size"] = self.env.action_size
        self.config["env_config"]["observation_size"] = self.env.observation_size
        self.config["env_config"]["episode_length"] = self.env.episode_length
        return self.env

    def setup_keys(self):
        key = jax.random.PRNGKey(self.config["exp_config"]["seed"])

        self.env_key, self.trainer_key, self.model_key = jax.random.split(key, 3)

    def setup_misc(self):

        with open("scripts/train_examples/tensorneat/default_config.yaml", "r") as f:
            self.default_config = yaml.load(f, Loader=yaml.SafeLoader)


    def get_final_policy(self):
        return self.training_info["policy_network"]["final"]
    def save_params(self, state):

        def callback(state):
            params, current_task = state
            current_task = current_task-1
            with open(self.config["exp_config"]["trial_dir"] + "/data/train/checkpoints/params_task_" + str(current_task) + ".pkl",
                      "wb") as f:
                pickle.dump(params, f)

        jax.debug.callback(callback, state)
        return None

    def init_run(self):

        # initialize model
        self.init_model()
        params, statics = eqx.partition(self.model, eqx.is_array)
        self.params_shaper = evosax.ParameterReshaper(params)

        def data_fn(data: dict):
            return {}

        # initialize environment



        self.pipeline = Pipeline(
            algorithm=self.model,
            problem=self.env,
            save_params_fn=self.save_params,
            generation_limit=self.config["optimizer_config"]["optimizer_params"]["generations"],
            seed=self.config["exp_config"]["trial_seed"],
            fitness_target=self.env.fitness_target,
            logger_run=self.logger_run
        )





    def cleanup(self):
        pass

    def run_trial(self):
        self.init_run()

        # print(state)
        # run until terminate
        state = self.pipeline.setup()
        state, best = self.pipeline.auto_run(state)

        self.final_state = {"state": state,
                            "params": best,
                            "algorithm": self.model,
                            "env_params": state.current_task }


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



    def viz_population(self):
        self.viz_policies()
        self.viz_performance()



    def save_training_info(self):



        super().save_training_info()

        with open(self.config["exp_config"]["trial_dir"] + "/data/train/final_training_state.pkl", "wb") as f:
            pickle.dump(self.final_state, f)

        #self.viz_population()

