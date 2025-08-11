
import functools
import os
import envs
import pickle
from scripts.train.rl.ppo.hyperparams import hyperparams
from methods.brax_wrapper.ppo import train as ppo
from scripts.train.base.experiment import Experiment
from functools import partial
import numpy as onp
import jax.numpy as jnp
import jax
from scripts.train.base.visuals import viz_histogram, viz_heatmap
from scripts.train.rl.ppo.hyperparams import hyperparams
from scripts.train.base.utils import max_rewards
from stepping_gates import envs as stepping_gates_envs
from brax import envs as brax_envs
from ecorobot import envs as ecorobot_envs
from envs.stepping_gates.stepping_gates.envs.wrappers import wrap as dgates_wrap
import wandb

# Register the hunted environment manually since it was removed from brax registry
import sys
sys.path.insert(0, '/home/eleni/workspace/old/neuroevolution_in_transfer_learning/.venv/lib/python3.12/site-packages/brax/envs')
from hunted import Hunted
brax_envs.register_environment('hunted', Hunted)

def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)

class PPOExperiment(Experiment):

    def __init__(self, env_config, model_config, exp_config, optimizer_config):
        super().__init__(env_config, model_config, exp_config, optimizer_config)
        


    def init_model(self):

        train_fn = functools.partial(ppo,
                                     **self.config["model_config"]["model_params"],
                                     **hyperparams[(self.config["env_config"]["env_name"])],
                                     episode_length=self.config["env_config"]["episode_length"],
                                     num_timesteps=self.config["optimizer_config"]["optimizer_params"]["num_timesteps"],
                                     seed=self.config["exp_config"]["trial_seed"])
        self.model = functools.partial(train_fn.func, **train_fn.keywords)

    def cleanup(self):
        pass
    
    
    def setup_stepping_gates_env(self):
        self.env = stepping_gates_envs.get_environment(env_name=self.config["env_config"]["env_name"],
                                                      **self.config["env_config"]["env_params"])
        self.config["env_config"]["action_size"] = self.env.action_size
        self.config["env_config"]["observation_size"] = self.env.observation_size
        self.config["env_config"]["episode_length"] = self.env.episode_length
        self.config["env_config"]["num_tasks"] = self.env.num_tasks
        
        
    def setup_brax_env(self):
        self.env = brax_envs.get_environment(env_name=self.config["env_config"]["env_name"],
                                                      **self.config["env_config"]["env_params"],
                                                      backend="mjx")
        self.env.reward_for_solved = max_rewards[self.config["env_config"]["env_name"]]
        self.env.num_tasks = 1
        self.config["env_config"]["action_size"] = self.env.action_size
        self.config["env_config"]["observation_size"] = self.env.observation_size
        self.config["env_config"]["episode_length"] = 1000
        self.config["env_config"]["num_tasks"] = 1
        
        
    def setup_ecorobot_env(self):
        self.env = ecorobot_envs.get_environment(env_name=self.config["env_config"]["env_name"],
                                                      **self.config["env_config"]["env_params"])
        self.env.reward_for_solved = max_rewards[self.config["env_config"]["env_name"]]
        self.env.num_tasks = 1
        self.config["env_config"]["action_size"] = self.env.action_size
        self.config["env_config"]["observation_size"] = self.env.observation_size
        self.config["env_config"]["episode_length"] = 1000
        self.config["env_config"]["num_tasks"] = 1

    def save_params(self, training_state):

        def callback(training_state):
            env_params = training_state.env_params
            current_task = jnp.ravel(env_params).astype(jnp.int32)[0]
            weights = self.params_to_weights(jax.tree_util.tree_map(lambda x: x[0,...], training_state.params.policy))
            with open(self.config["exp_config"]["trial_dir"] + "/data/train/checkpoints/params_task_" + str(current_task) + ".pkl",
                      "wb") as f:
                pickle.dump(( _unpmap(training_state.env_steps),
                             _unpmap(training_state.normalizer_params),
                             weights,
                             _unpmap(training_state.params.policy)), f)

        jax.debug.callback(callback, training_state)


    def progress(self, info):
        gen, env_params, metrics = info

        def log(wandb_info):
            logging_info = {
                "current_best_fitness": wandb_info["fitness"],
                "generation": wandb_info["gen"],
                "current_task": wandb_info["current_task"]
            }
            wandb.log(logging_info)

        total_eval_info = {
            "fitness": metrics["eval/episode_reward"],
            "gen": gen,
            "current_task": env_params[0][0]}
        log(total_eval_info)
        
        print("current best fitness: ", total_eval_info["fitness"])
        print("current task: ", total_eval_info["current_task"])

    def eval_task(self, policy_params, tasks, gens, final_policy=False):
        inference_fn = self.final_state["inference_fn"](policy_params)

        act_fn = partial(jax.jit(inference_fn), key_sample=jax.random.PRNGKey(0))
        super().run_eval(act_fn, tasks, final_policy)

   

    def train_trial(self):
        self.init_model()

        make_inference_fn, params, _, training_state = self.model(environment=self.env,
                                                                  progress_fn=self.progress,
                                                                  save_params_fn=self.save_params)

        self.final_state = {"inference_fn": make_inference_fn,
                            "params": params,
                            "env_params": training_state.env_params,
                            }


    def params_to_weights(self, params):
        width = 0
        height = 0
        kernels = []
        for el in params["params"].values():
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
        for el in params["params"].values():
            kernel = el["bias"]
            width += kernel.shape[0]
        width = width + self.env.observation_size
        bias = jnp.zeros((width,))
        start_x = 0
        for el in params["params"].values():
            bias = bias.at[start_x:start_x + el["bias"].shape[0]].set(el["bias"])
            start_x += el["bias"].shape[0]

        bias = onp.array(bias)

        weights = onp.array(weights)
        weights = onp.vstack((bias[onp.newaxis, :], weights))
        #weights = onp.hstack((onp.zeros((width + 1, 1)), weights))
        return weights
    
    def get_params(self, f):
        gens, normalizer_params, weights_array, params = pickle.load(f)
        return gens, (normalizer_params, params)

    
    def save_training_info(self):

        # weights = self._map_to_adjacency_matrix(conns)
        params = self.final_state["params"][1]
        weights = self.params_to_weights(params)
    
        viz_heatmap(weights,
                    filename=self.config["exp_config"]["trial_dir"] + "/visuals/train/policy/final")
        
        num_tasks = len(os.listdir(self.config["exp_config"]["trial_dir"] + "/data/train/checkpoints"))

        checkpoint_weights = []
        for task in range(num_tasks):
            with open(self.config["exp_config"]["trial_dir"] + "/data/train/checkpoints/params_task_" + str(
                    task) + ".pkl","rb") as f:
                _, normalizer_params, weights_array, params = pickle.load(f)
                #task_weights = self.params_to_weights(params)
                checkpoint_weights.append(weights_array)
                
                viz_heatmap(weights_array,
                    filename=self.config["exp_config"]["trial_dir"] + "/visuals/train/policy/task_" + str(task))


        # save final policy matrix
        self.training_info = {"policy_network": {"final": weights,
                                                 "checkpoints": checkpoint_weights}}

        super().save_training_info()

