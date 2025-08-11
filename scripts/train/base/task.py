import jax
import os
from scripts.train.base.utils import max_rewards
import numpy as onp

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

        self.eval_info = {}

    def run_eval(self, act_fn, saving_dir, tasks, gens=None, final_policy=False):
        env = self.env
        jit_env_reset = jax.jit(env.reset)
        jit_env_step = jax.jit(env.step)


        for task in tasks:
            
            trial_rewards = []
            trial_success = []

            if not os.path.exists(saving_dir + "/task_" + str(task)):
                os.makedirs(saving_dir + "/task_" + str(task))

            if final_policy:
                obs_size = self.env.get_obs_size(self.num_tasks)
                action_size = self.env.get_action_size(self.num_tasks)

            else:
                obs_size = self.env.get_obs_size(task)
                action_size = self.env.get_action_size(task)

            for eval_trial in range(self.num_eval_trials):

                rng = jax.random.PRNGKey(seed=eval_trial)
                if self.config["env_config"]["env_name"] != "hunted":
                    state = jit_env_reset(rng, jax.numpy.array([task]))
                else:
                    state = jit_env_reset(rng)
                cum_reward = 0
                infos = []
                actions = []
                states = []
                success = 0
                episode_length = 0

                for step in range(self.config["env_config"]["episode_length"]):

                    prev_obs = state.obs

                    act_rng, rng = jax.random.split(rng)


                    if self.config["optimizer_config"]["optimizer_name"] == "ppo":
                        act = act_fn(prev_obs)

                    else:
                        act = act_fn(prev_obs, action_size=action_size, obs_size=obs_size)

                    if isinstance(act, tuple):
                        act, info = act
                        infos.append(info)

                    
                    state = jit_env_step(state, act)
                    reward = state.reward
                    done = state.done

                    cum_reward += float(reward)
                    actions.append(act)
                    if reward == max_rewards[self.config["env_config"]["env_name"]]:
                        success += 1
                    episode_length += 1

                    if self.config["env_config"]["env_type"] == "stepping_gates":
                        states.append(onp.array([["Timestep", "Observation", "Action", "Label"],
                                                 [str(step), str(prev_obs), str(act), str(state.info["label"])]]))
                    else:
                        states.append(state.pipeline_state)
                    if done:
                        break
                trial_success.append(float(success / episode_length))
                trial_rewards.append(float(cum_reward))

                gif_path = self.env.show_rollout(states, save_dir=saving_dir + "/task_" + str(task),
                                                     filename="eval_trial_" + str(eval_trial) + "_final_" + str(final_policy) + "_rew_" + str(cum_reward))

            task_alias = "task_" + str(task)
            if final_policy:
                task_alias += "_final_policy"

                self.eval_info[task_alias] = {"rewards": [float(el) for el in trial_rewards],
                                        "success": [float(el) for el in trial_success]}
            else:
                self.eval_info[task_alias] = {"rewards": [float(el) for el in trial_rewards],
                                        "success": [float(el) for el in trial_success],
                                        "gens": gens}
