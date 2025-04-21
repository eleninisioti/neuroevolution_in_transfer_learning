import jax.numpy as jnp

import stepping_gates.envs as envs

from .rl_jit import RLEnv

from typing import Callable

import jax
from jax import vmap, numpy as jnp
import numpy as np

from ..base import BaseProblem
from tensorneat.common import State


def norm_obs(state, obs):
    return (obs - state.problem_obs_mean) / (state.problem_obs_std + 1e-6)


class GatesEnv(RLEnv):
    def __init__(self, config):
        super().__init__()
        self.env = envs.create(env_name=config["env_name"], **config["env_params"])
        self.fitness_target = self.env.max_reward

        self.action_size = self.env.action_size
        self.observation_size = self.env.observation_size
        self.episode_length = self.env.episode_length
        self.n_input = self.env.n_input
        self.task_params = self.env.task_params
        self.num_tasks = self.env.num_tasks
        self.current_task = self.env.current_task


    def setup(self, state):
        state = super().setup(state)
        state = state.register(current_task=self.env.current_task)
        return state

    def step(self, env_state, action):
        return self.env_step( env_state, action)

    def update_env_params(self, env_params):
        return self.env.update_env_params(env_params)


    def get_obs_size(self, task):
        return self.env.get_obs_size(task)

    def get_action_size(self, task):
        return self.env.get_action_size(task)
    def env_step(self, env_state, action):
        state = self.env.step(env_state, action)
        return state

    def env_reset(self, randkey):
        init_state = self.env.reset(randkey)
        return init_state.obs, init_state


    def _evaluate_once(
        self,
        state,
        randkey,
        act_func,
        params,
        action_policy,
        record_episode,
        normalize_obs=False,
    ):
        rng_reset, rng_episode = jax.random.split(randkey)
        init_env_state = self.reset(rng_reset, state.current_task)
        init_obs = init_env_state.obs

        if record_episode:
            obs_array = jnp.full((self.max_step, *self.input_shape), jnp.nan)
            action_array = jnp.full((self.max_step, *self.output_shape), jnp.nan)
            reward_array = jnp.full((self.max_step,), jnp.nan)
            episode = {
                "obs": obs_array,
                "action": action_array,
                "reward": reward_array,
            }
        else:
            episode = None

        def cond_func(carry):
            _, _, _, done, _, count, _, rk = carry
            return ~done & (count < self.max_step)

        def body_func(carry):
            (
                obs,
                env_state,
                rng,
                done,
                tr,
                count,
                epis,
                rk,
            ) = carry  # tr -> total reward; rk -> randkey

            if normalize_obs:
                obs = norm_obs(state, obs)

            if action_policy is not None:
                forward_func = lambda obs: act_func(state, params, obs)
                action = action_policy(rk, forward_func, obs)
            else:
                action = act_func(state, params, obs)
            next_state= self.step(
                 env_state, action
            )
            next_obs = next_state.obs
            next_env_state = next_state
            reward = next_state.reward
            done = next_state.done
            next_rng, _ = jax.random.split(rng)

            if record_episode:
                epis["obs"] = epis["obs"].at[count].set(obs)
                epis["action"] = epis["action"].at[count].set(action)
                epis["reward"] = epis["reward"].at[count].set(reward)

            return (
                next_obs,
                next_env_state,
                next_rng,
                done,
                tr + reward,
                count + 1,
                epis,
                jax.random.split(rk)[0],
            )

        _, _, _, _, total_reward, _, episode, _ = jax.lax.while_loop(
            cond_func,
            body_func,
            (init_obs, init_env_state, rng_episode, False, 0.0, 0, episode, randkey),
        )
        total_reward = total_reward/self.env.episode_length


        def callback(state):
            params, current_task = state
            current_task = current_task-1
            return None

        #jax.lax.cond(total_reward==0, jax.debug.callback(callback, state), )


        if record_episode:
            return total_reward, episode
        else:
            return total_reward
    def reset(self, key, env_params):
        return self.env.reset(key, env_params)


    def render(self, state, action):
        return self.env.render(state, action)

    def show_rollout(self, states, saving_dir, filename):
        return self.env.show_rollout(states, saving_dir, filename)

    @property
    def input_shape(self):
        return (self.env.observation_size,)

    @property
    def output_shape(self):
        return (self.env.action_size,)

    def show(self, randkey, state, act_func, params, save_path=None, height=512, width=512, duration=0.1, *args, **kwargs):

        import jax

        obs, env_state = self.reset(randkey)
        reward, done = 0.0, False
        state_histories = []

        def step(key, env_state, obs):
            key, _ = jax.random.split(key)
            action = act_func( obs, params)
            obs, env_state, r, done, _ = self.step(randkey, env_state, action)
            return key, env_state, obs, r, done, action

        states = []
        actions = []
        rewards = []
        while not done:
            state_histories.append(env_state)
            key, env_state, obs, r, done, action = jax.jit(step)(randkey, env_state, obs)

            states.append(self.render(env_state, action))
            reward += r
            actions.append(str(action))
            rewards.append(str(r))
        reward = jnp.mean(rewards)

        """
        file_path = save_path
        with open(file_path, 'a') as file:
            # Write each string to the file
            file.write("States: " + ' '.join(map(str, states)) + "\n")
            file.write("Actions: " + ' '.join(map(str, actions)) + "\n")
            file.write("Rewards: " + ' '.join(map(str, rewards)) + "\n")
        print("Trajectory saved to: ", save_path)
        print("Total reward: ", reward)
        """

