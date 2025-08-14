import jax
import os
from scripts.train.base.utils import max_rewards
import numpy as onp
import jax.numpy as jnp

class Task:

    def __init__(self, env, config):
        self.env = env
        self.config = config

        if config["env_config"]["curriculum"]:
            self.num_tasks = env.num_tasks
            self.current_task = env.current_task
        else:
            self.num_tasks = 1
            self.current_task = 0

        if config["env_config"]["env_type"] == "stepping_gates":
            if config["optimizer_config"]["optimizer_type"] == "tensorneat":
                self.num_eval_trials = 1
            elif config["optimizer_config"]["optimizer_type"] == "evosax":
                self.num_eval_trials = 1
            elif config["optimizer_config"]["optimizer_type"] == "brax":
                self.num_eval_trials = 100

        elif config["env_config"]["env_type"] == "brax":
            self.num_eval_trials = 10

        elif config["env_config"]["env_type"] == "ecorobot":
            self.num_eval_trials = 100

        elif config["env_config"]["env_type"] == "gymnax":
            self.num_eval_trials = 10

        elif config["env_config"]["env_type"] == "pattern_match":
            self.num_eval_trials = 1
            
        self.env_type = config["env_config"]["env_type"]

        self.eval_info = {}
        
        
    def get_input_ouput(self, task):
        obs_size = self.config["env_config"]["observation_size"]
        action_size = self.config["env_config"]["action_size"]

            
        return obs_size, action_size
    
    def run_eval_trial_gymnax(self, env, task, eval_trial, act_fn, obs_size, action_size):
        trial_rewards = []
        trial_success = []
        jit_env_reset = jax.jit(env.reset)
        jit_env_step = jax.jit(env.step)
        
        rng = jax.random.PRNGKey(seed=eval_trial)
        obs, state = jit_env_reset(rng, jax.numpy.array([task]))
        cum_reward = 0
        infos = []
        actions = []
        states = []
        success = 0
        episode_length = 0

        for step in range(self.config["env_config"]["episode_length"]):

            prev_obs = obs

            act_rng, rng = jax.random.split(rng)
            
            
            if self.config["optimizer_config"]["optimizer_name"] == "ppo":
                act = act_fn(prev_obs)
            else:
                act = act_fn(prev_obs, action_size=action_size, obs_size=obs_size)


            #act = act_fn(prev_obs, action_size=action_size, obs_size=obs_size)
            if isinstance(act, tuple):
                act, info = act
                infos.append(info)
            act = jnp.argmax(act)

            obs, state, reward, done, _ = jit_env_step(act_rng, state, act, self.config["env_config"]["gymnax_env_params"])

            cum_reward += float(reward)
            actions.append(act)
            if reward == max_rewards[self.config["env_config"]["env_name"]]:
                success += 1
            episode_length += 1

            if done:
                break
        trial_success.append(float(success / episode_length))
        trial_rewards.append(float(cum_reward))
        
        return trial_rewards, trial_success    
    
    def run_eval_trial(self, env, task, eval_trial, act_fn, obs_size, action_size):
        if self.env_type == "gymnax":
            return self.run_eval_trial_gymnax(env, task, eval_trial, act_fn, obs_size, action_size)
        else:
            return self.run_eval_trial_brax(task, env, eval_trial, act_fn, obs_size, action_size)
        
        
        
        
        
        

    def run_eval(self, act_fn, saving_dir, tasks, gens=None, final_policy=False):
        env = self.env

        for task in tasks:
            


            if not os.path.exists(saving_dir + "/task_" + str(task)):
                os.makedirs(saving_dir + "/task_" + str(task))

            if final_policy:
                obs_size, action_size = self.get_input_ouput(self.num_tasks)


            else:
                obs_size, action_size = self.get_input_ouput(task)
                
            

            for eval_trial in range(self.num_eval_trials):
                
                trial_rewards, trial_success= self.run_eval_trial(env, task, eval_trial, act_fn, obs_size, action_size)


            task_alias = "task_" + str(task)
            if final_policy:
                task_alias += "_final_policy"

                self.eval_info[task_alias] = {"rewards": [float(el) for el in trial_rewards],
                                        "success": [float(el) for el in trial_success]}
            else:
                self.eval_info[task_alias] = {"rewards": [float(el) for el in trial_rewards],
                                        "success": [float(el) for el in trial_success],
                                        "gens": gens}
