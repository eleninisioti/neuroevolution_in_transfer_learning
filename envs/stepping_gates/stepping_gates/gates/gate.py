
import jax.numpy as jnp
import jax
from gym import spaces

class Gate:

    def __init__(self, n_control_bits, n_output_circuit):
        self.n_control_bits = n_control_bits
        self.n_output_circuit = n_output_circuit


        self.observation_space = spaces.MultiDiscrete([2 for i in range(self.n_input)])

    def get_n_inactive_input(self):
        return self.n_inactive_input

    def set_inactive_input(self, n_input_circuit):
        self.n_inactive_input = n_input_circuit - self.n_input
        self.inactive_input = jnp.array([0]*self.n_inactive_input, dtype=jnp.int32) # we need to set the bits that the gate is not using

    def reset(self, key):
        pass

    def sample_multi_discrete(self, key):
        # Convert the nvec (number of values per dimension) to a JAX array
        nvec = jnp.array(self.observation_space.nvec)
        # Generate random integers in the range [0, n-1] for each dimension
        return jax.random.randint(key, shape=nvec.shape, minval=0, maxval=nvec)

    def reset_random(self, key):
        return  jnp.concatenate([self.inactive_input, self.sample_multi_discrete(key)])




    def get_inactive_outputs(self):

        inactive_output = jnp.zeros((self.n_output_circuit,))
        inactive_output = inactive_output.at[-self.n_inactive_output:].set(1)

        inactive_output = jnp.where(self.n_inactive_output, inactive_output, jnp.zeros_like(inactive_output))
        return inactive_output

    def preprocess_action(self, action):
        action_active =action
        #action_active = jnp.where(jnp.greater(jax.nn.sigmoid(action_active), 0.5), jnp.ones(action_active.shape), jnp.zeros(action_active.shape))
        action_active = jnp.where(jnp.greater(action_active,0), 1, 0)
        #return jnp.concatenate([action_active, jnp.ones(action[self.n_output:].shape)])

        return action_active

    def increment(self, binary_input):
        bin_int = jnp.sum(binary_input * 2 ** jnp.arange(len(binary_input) - 1, -1, -1))
        incremented_value = bin_int + 1

        # Convert incremented integer to binary representation as a JAX array
        bin_str_length = len(binary_input)
        incremented_bin_arr = jnp.array([(incremented_value >> i) & 1 for i in range(bin_str_length - 1, -1, -1)])

        return incremented_bin_arr.astype(jnp.int32)

    def step(self, state):
        current_step = state.obs[self.n_control_bits+self.n_inactive_input:self.n_control_bits+self.n_inactive_input + self.n_input]
        next_step = self.increment(current_step)
        new_obs = jnp.concatenate([self.inactive_input, next_step])
        #new_obs = next_step

        return new_obs

    def add_inactive_output(self, label):
        return jnp.concatenate([label, self.inactive_output])

    def function(self, state):
        pass