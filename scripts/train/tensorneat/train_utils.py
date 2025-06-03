import sys

import matplotlib.pyplot as plt

import envs
from scripts.train.base.experiment import Experiment
import yaml
import jax
import equinox as eqx
import jax.numpy as jnp
import pickle
import os
from  methods.tensorneat.pipeline import Pipeline
from methods.tensorneat.problem.func_fit.n_parity import Nparity
from  methods.tensorneat.problem.rl.gates_env import GatesEnv
import numpy as onp
from collections import defaultdict
import json
import pandas as pd
from functools import partial


class TensorneatExperiment(Experiment):

    def __init__(self, env_config, model_config, exp_config, optimizer_config):
        super().__init__(env_config, model_config, exp_config, optimizer_config)


    def setup_stepping_gates_env(self):

        self.env = GatesEnv(self.config["env_config"])


        self.config["env_config"]["action_size"] = self.env.action_size
        self.config["env_config"]["observation_size"] = self.env.observation_size
        self.config["env_config"]["episode_length"] = self.env.episode_length
        self.config["env_config"]["num_tasks"] = self.env.num_tasks


        return self.env


    def setup_trial_keys(self):
        key = jax.random.PRNGKey(self.config["exp_config"]["trial_seed"])
        self.env_key, self.trainer_key, self.model_key = jax.random.split(key, 3)


    def get_final_policy(self):
        return self.training_info["policy_network"]["final"]
    
    def save_params(self, state):

        def callback(state):
            params, current_task, gen = state
            current_task = current_task-1
            with open(self.config["exp_config"]["trial_dir"] + "/data/train/checkpoints/params_task_" + str(current_task)+ ".pkl",
                      "wb") as f:
                pickle.dump((gen, params), f)

        jax.debug.callback(callback, state)
        return None

    def init_run(self):

        # initialize model
        self.init_model()

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
        )



    def cleanup(self):
        pass

    def train_trial(self):
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




    def eval_task(self, policy_params, tasks, gens=None, final_policy=False):
        pop_transformed = self.load_model(policy_params)
        state = self.final_state["state"]

        # Fix `y` and `z`
        # act_fn = partial(self.model.forward, state=state, transformed=pop_transformed)

        def act_fn(obs, action_size=None, obs_size=None):
            return self.model.forward(state=state, transformed=pop_transformed, inputs=obs)

        super().run_eval(jax.jit(act_fn), tasks, gens=gens, final_policy=final_policy)




    def save_training_info(self):

        super().save_training_info()

        with open(self.config["exp_config"]["trial_dir"] + "/data/train/final_training_state.pkl", "wb") as f:
            pickle.dump(self.final_state, f)

        #self.viz_population()

