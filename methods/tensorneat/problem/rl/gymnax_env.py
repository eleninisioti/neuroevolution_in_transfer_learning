import gymnax

from .rl_jit import RLEnv
import jax.numpy as jnp
import jax
class GymnaxEnv(RLEnv):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env, self.env_params = gymnax.make(env_id=config["env_name"])

        if config["env_name"] == "CartPole-v1":

            self.action_size = 1
            self.observation_size = self.env.observation_space(self.env_params).shape[0]
            self.fitness_target = 500

        else:
            self.action_size = 3
            self.observation_size = self.env.observation_space(self.env_params).shape[0]
            self.fitness_target = -60
        self.episode_length = 500
        self.current_task = 0
        self.num_tasks = 1

    def get_obs_size(self, task):
        return self.observation_size

    def get_action_size(self, task):
        return self.action_size

    def setup(self, state):
        state = super().setup(state)
        state = state.register(current_task=self.current_task)
        return state

    def env_step(self, randkey, env_state, action):
        if self.env.name == "CartPole-v1":
            action = jnp.where(action > 0.5, 1, 0)[0]
        else:
            action = jnp.argmax(action)
        return self.env.step(randkey, env_state, action, self.env_params)

    def reset(self, key, env_params):
        return self.env.reset(key)
    def env_reset(self, randkey):
        return self.env.reset(randkey, self.env_params)

    def step(self, env_state, action):
        randkey = jax.random.PRNGKey(seed=0)
        return self.env_step(randkey, env_state, action)

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
        init_obs, init_env_state = self.reset(rng_reset, state.current_task)

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


            if action_policy is not None:
                forward_func = lambda obs: act_func(state, params, obs)
                action = action_policy(rk, forward_func, obs)
            else:
                action = act_func(state, params, obs)

            next_obs, next_state, reward, done, _= self.step(
                 env_state, action
            )
            next_env_state = next_state
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
        total_reward = total_reward


        def callback(state):
            params, current_task = state
            current_task = current_task-1
            return None

        #jax.lax.cond(total_reward==0, jax.debug.callback(callback, state), )


        if record_episode:
            return total_reward, episode
        else:
            return total_reward

    @property
    def input_shape(self):
        return self.env.observation_space(self.env_params).shape

    @property
    def output_shape(self):
        return self.env.action_space(self.env_params).shape

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        raise NotImplementedError
