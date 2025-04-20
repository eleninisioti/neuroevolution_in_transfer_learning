from gym import spaces
import jax
import jax.numpy as jnp
from stepping_gates.envs.base import Circuit, State
from stepping_gates.gates import Parity


class NParity(Circuit):
    """
    This task requires computing the parity of a binary set. If the set has an even number of ones the correct action is 0, if it has an odd number of ones it is 1.

    It can be customized in 3 ways:

    - controlling the number of input bits N
    - choosing between two modes: 'one-step' mode has episodes of length 1 where a random observation
     is sampled (useful for RL agents like PPO) while 'full' mode goes through all possible observations in a single episode.
    - activating a curriculum. When 'curriculum' is False the task is as described before. When 'curriculum' is True then there N-2 tasks, where the first task
    requires finding the parity of the first two input bits, the second task of the first three input bits, the third of the first four inputs bits etc.
    The current task is set through the reset method.

    Rewards:
    At each step of an episode the reward is -(correct_output - actions)**2. Thus the optimal reward is 0 (for both modes).

    """

    def __init__(self, n_input=6, curriculum=False, episode_type="one-step" ):

        super().__init__()

        self.n_input = n_input
        self.n_output = 1

        self.action_space = spaces.MultiDiscrete([2 for _ in range(self.n_output)])
        self.observation_space = spaces.MultiDiscrete([2 for _ in range(self.n_input)])

        if episode_type == "one_step":
            self.episode_length = 1
        else:
            self.episode_length = 2 ** self.n_input
        self.episode_type = episode_type

        self.max_reward = 0.0
        self.reward_for_solved =0
        if curriculum:
            self.task_params = jnp.array([2, 3, 4, 5, 6]) # task 0 has 2 inputs, task 1 has 3 ....
            self.num_tasks = len(self.task_params)
        else:
            self.num_tasks = 1
            self.task_params = jnp.array([6]) # task 0 has 2 inputs, task 1 has 3 ....
        self.current_task = 0

        self.gate = Parity(n_input_circuit=self.n_input, n_output_circuit=self.n_output)

    def sample_multi_discrete(self, key):
        # Convert the nvec (number of values per dimension) to a JAX array
        nvec = jnp.array(self.observation_space.nvec)
        # Generate random integers in the range [0, n-1] for each dimension
        return jax.random.randint(key, shape=nvec.shape, minval=0, maxval=nvec)



    def deactivate_obs(self, obs, n_active_inputs):
        return jnp.where(jnp.arange(self.n_input) >= n_active_inputs, 0, obs)


    def get_obs_size(self, task):
        return self.task_params[task]

    def get_action_size(self, task):
        return 1

    def reset(self, key, env_params=jnp.array([0])):

        env_params = jnp.ravel(env_params).astype(jnp.int32)[0]

        env_params = jnp.where(env_params >= self.num_tasks, self.num_tasks-1, env_params)
        n_active_inputs = self.task_params[env_params]

        if self.episode_type == "one-step":
            obs = self.sample_multi_discrete(key)
            episode_length = 1
        else:
            obs = jnp.zeros((self.n_input,)).astype(jnp.int32)
            episode_length =  2**n_active_inputs

        info = {"steps": 0.0,
                "truncation": 0.0, # this does nothing for the env but is required by brax ppo
                "label": 0,
                "action": jnp.array([0]),
                "current_task": env_params,
                "episode_length": episode_length
                }


        obs = jnp.where(jnp.arange(self.n_input) >= (self.n_input- n_active_inputs), obs, 0)
        return State(obs=obs, done=False, reward=0.0, info=info, metrics={"reward": 0.0})


    def increment(self, binary_input):
        bin_int = jnp.sum(binary_input * 2 ** jnp.arange(len(binary_input) - 1, -1, -1))
        incremented_value = bin_int + 1

        # Convert incremented integer to binary representation as a JAX array
        bin_str_length = len(binary_input)
        incremented_bin_arr = jnp.array([(incremented_value >> i) & 1 for i in range(bin_str_length - 1, -1, -1)])

        return incremented_bin_arr.astype(jnp.int32)


    def step(self, state, action):
        obs = state.obs
        action = jnp.where(jnp.greater(action,0.0), 1, 0)[0]
        n_active_inputs = self.task_params[state.info["current_task"]]

        obs = jnp.where(jnp.arange(self.n_input) >= (self.n_input- n_active_inputs), obs,0) # to correctly compute the label
        label = self.gate.function(obs)
        new_obs = self.increment(obs)

        new_obs = jnp.where(jnp.arange(self.n_input) >= (self.n_input- n_active_inputs),new_obs,0)

        reward = (-(label-action)**2).astype(jnp.float32)

        state.info["steps"] = state.info["steps"] + 1
        done = jnp.where(state.info["steps"]==state.info["episode_length"], True, False)
        done = jnp.where(state.done, True, done)

        state.info["label"] = label
        state.info["action"] = jnp.array([action])
        state.metrics.update(reward=reward)
        new_state = State(reward=reward, done=done,obs=new_obs, info=state.info, metrics=state.metrics)
        return new_state

    def render(self, state, action, mode='human'):
        # Render the environment to the screen (optional)
        rollout = "Observation: " + str(state.obs) + "\n"
        rollout += "Action: " + str(action) + "\n"
        rollout += "Reward: " + str(state.reward) + "\n"
        rollout += "Done: " + str(state.done) + "\n"
        return rollout



