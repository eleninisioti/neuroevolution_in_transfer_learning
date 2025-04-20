from gym import spaces
import jax
import jax.numpy as jnp
from stepping_gates.envs.base import Circuit, State
from stepping_gates.gates import NAND, NOT, AND, XOR, XAX, OR, Incr, Decr, Shiftleft, Multiplexer
import numpy as onp
import itertools


class SimpleALU(Circuit):

    def __init__(self,  episode_type="one_step", curriculum=False, reward_for_solved=None):
        super().__init__()

        self.n_input = 4
        self.n_output = 4
        self.n_input_control = 4

        self.action_space = spaces.MultiDiscrete([3 for i in range(self.n_output)])
        self.observation_space = spaces.MultiDiscrete([2 for i in range(self.n_input_control)] + [2 for i in range(self.n_input)])

        self.max_reward = 0.0
        self.reward_for_solved = 0.0


        gates = [
            Multiplexer(n_input_circuit=self.n_input, n_output_circuit=self.n_output, n_control_bits=self.n_input_control),
            NAND(n_input_circuit=self.n_input, n_output_circuit=self.n_output, n_control_bits=self.n_input_control),
            NOT(n_input_circuit=self.n_input, n_output_circuit=self.n_output, n_control_bits=self.n_input_control),
            AND(n_input_circuit=self.n_input, n_output_circuit=self.n_output,
                n_control_bits=self.n_input_control),
            #OR(n_input_circuit=self.n_input, n_output_circuit=self.n_output, n_control_bits=self.n_input_control),
            XOR(n_input_circuit=self.n_input, n_output_circuit=self.n_output, n_control_bits=self.n_input_control),
            XAX(n_input_circuit=self.n_input, n_output_circuit=self.n_output, n_control_bits=self.n_input_control),
                 Incr(n_input_circuit=self.n_input, n_output_circuit=self.n_output, n_control_bits=self.n_input_control),
                 Decr(n_input_circuit=self.n_input, n_output_circuit=self.n_output, n_control_bits=self.n_input_control),
                 Shiftleft(n_input_circuit=self.n_input, n_output_circuit=self.n_output, n_control_bits=self.n_input_control),
                 ]
        if episode_type == "one_step":
            self.episode_length = 1
            self.episode_lengths = jnp.array([1 for el in range(len(gates))])
        else:
            self.episode_lengths = onp.array([gate.episode_length for gate in gates])
            self.episode_length = int(onp.sum(self.episode_lengths))
        self.episode_type = episode_type

        if curriculum:
            self.task_params = jnp.arange(len(gates))
        else:
            self.task_params = jnp.array([len(gates)-1])
        self.current_task = 0

        self.num_tasks = len(self.task_params)
        self.steps_function = [el.function for el in gates]
        self.steps_step = [el.step for el in gates]
        self.steps_reset = [el.reset for el in gates]
        self.steps_preprocess = [el.preprocess_action for el in gates]
        self.steps_get_inactive_outputs = [el.get_inactive_outputs for el in gates]
        self.steps_n_inactive_inputs = [el.get_n_inactive_input for el in gates]

    def int_to_binary_array(self,x):
        x = jnp.array([x])
        # Create a binary mask for each bit position
        mask = jnp.array([8, 4, 2, 1])  # 8 = 2^3, 4 = 2^2, 2 = 2^1, 1 = 2^0
        # mask = jnp.array([2,1])  # 8 = 2^3, 4 = 2^2, 2 = 2^1, 1 = 2^0
        # temp = x.shape[0]
        # mask = mask[-x.shape[0]:]
        # Perform bitwise AND to extract each bit
        binary_rep = (x[:, None] & mask) > 0

        return jnp.squeeze(binary_rep.astype(jnp.int32))


    def sample_multi_discrete(self, key):
        # Convert the nvec (number of values per dimension) to a JAX array
        nvec = jnp.array(self.observation_space.nvec)
        # Generate random integers in the range [0, n-1] for each dimension
        return jax.random.randint(key, shape=nvec.shape, minval=0, maxval=nvec)



    def deactivate_obs(self, obs, n_active_inputs):
        return jnp.where(jnp.arange(self.n_input) >= n_active_inputs, 0, obs)



    def reset(self, key, env_params=jnp.array([0])):

        env_params = jnp.ravel(env_params).astype(jnp.int32)[0]
        current_function = self.task_params[env_params.astype(jnp.int32)]

        episode_length = jnp.take(self.episode_lengths, current_function)

        info = {"steps": 0.0,
                "truncation": 0.0, # this does nothing for the env but is required by brax ppo
                "label": jnp.array([0,0,0,0]),
                "action": jnp.zeros((1,self.n_output)).astype(jnp.int32),
                "current_task": current_function,
                "episode_length": episode_length
                }
        if self.episode_type == "one_step":
            obs = self.sample_multi_discrete(key)
            obs = obs[self.n_input_control:]
        else:
            obs = jnp.zeros((self.n_input,)).astype(jnp.int32)

        n_inactive_inputs = jax.lax.switch(current_function,self.steps_n_inactive_inputs)
        obs = jnp.where(jnp.arange(self.n_input) < n_inactive_inputs, -1, obs)
        control_bits = self.int_to_binary_array(current_function)
        obs = jnp.concatenate([control_bits, obs])
        return State(obs=obs, done=False, reward=0.0, info=info, metrics={"reward": 0.0})

    def increment(self, binary_input):
        bin_int = jnp.sum(binary_input * 2 ** jnp.arange(len(binary_input) - 1, -1, -1))
        incremented_value = bin_int + 1

        # Convert incremented integer to binary representation as a JAX array
        bin_str_length = len(binary_input)
        incremented_bin_arr = jnp.array([(incremented_value >> i) & 1 for i in range(bin_str_length - 1, -1, -1)])

        return incremented_bin_arr.astype(jnp.int32)

    def function(self, state):
        label = jnp.logical_not(jnp.logical_and(state[0], state[1]))
        return label

    def step(self, state, action):
        obs = state.obs
        action = jnp.where(jnp.greater(action,0.0), 1, 0)

        label = jax.lax.switch(state.info["current_task"],  self.steps_function, obs[self.n_input_control:])

        new_obs = jax.lax.switch(state.info["current_task"],  self.steps_step, state)

        control_bits = self.int_to_binary_array(state.info["current_task"])
        new_obs = jnp.concatenate([control_bits, new_obs])

        match = jnp.where(jnp.logical_or(label==2, label==action), 1, 0)
        #match = jnp.all(match==1)
        reward = -jnp.sum((1-match)**2).astype(jnp.float32)/self.n_output

        state.info["steps"] = state.info["steps"] + 1
        done = jnp.where(state.info["steps"]==state.info["episode_length"], True, False)

        state.info["label"] = label
        state.info["action"] = jnp.array([action])
        state.metrics.update(reward=reward)
        done = jnp.where(state.done, True, done)

        new_state = State(reward=reward, done=done,obs=new_obs, info=state.info, metrics=state.metrics)
        return new_state


    def render(self, state, action, mode='human'):
        # Render the environment to the screen (optional)
        rollout = "Observation: " + str(state.obs) + "\n"
        rollout += "Action: " + str(action) + "\n"
        rollout += "Reward: " + str(state.reward) + "\n"
        rollout += "Done: " + str(state.done) + "\n"
        return rollout