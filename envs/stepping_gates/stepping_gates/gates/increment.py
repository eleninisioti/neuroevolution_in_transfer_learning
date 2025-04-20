import jax.numpy as jnp
from stepping_gates.gates.gate import Gate
from stepping_gates.gates.utils import *
import jax

class Incr(Gate):

    def __init__(self, n_input_circuit, n_output_circuit, n_control_bits):

        self.episode_length = 16

        self.n_input = 4
        self.n_output = 4

        super().__init__(n_control_bits, n_output_circuit)


        super().set_inactive_input(n_input_circuit)

        self.n_inactive_output = n_output_circuit - self.n_output
        self.inactive_output = jnp.array([2] * self.n_inactive_output,
                                        dtype=jnp.int32)  # we need to set the bits that the gate is not using


    def reset(self, key):
            #temp =jnp.concatenate([jnp.array([0, 0, 0, 0]), self.inactive_input])
            return jnp.concatenate([self.inactive_input, jnp.array([0, 0,0,0]) ])



    def get_inactive_outputs(self):

        inactive_output = jnp.zeros((self.n_output_circuit,))
        return inactive_output


    def function(self, binary_input):
        bin_int = jnp.sum(binary_input * 2 ** jnp.arange(len(binary_input) - 1, -1, -1))
        incremented_value = bin_int + 1

        # Convert incremented integer to binary representation as a JAX array
        bin_str_length = len(binary_input)
        incremented_bin_arr = jnp.array([(incremented_value >> i) & 1 for i in range(bin_str_length - 1, -1, -1)])

        label = self.add_inactive_output(incremented_bin_arr)

        return label





