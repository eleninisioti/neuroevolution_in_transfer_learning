import jax.numpy as jnp
from stepping_gates.gates.gate import Gate

class NOT(Gate):

    def __init__(self, n_input_circuit, n_output_circuit, n_control_bits):

        self.episode_length = 2

        self.n_input = 1
        self.n_output = 1
        super().__init__(n_control_bits, n_output_circuit)


        super().set_inactive_input(n_input_circuit)

        self.n_inactive_output = n_output_circuit - self.n_output
        self.inactive_output = jnp.array([2] * self.n_inactive_output,
                                        dtype=jnp.int32)  # we need to set the bits that the gate is not using


    def reset(self, key):
        return jnp.concatenate([self.inactive_input, jnp.array([0]), ])


    def function(self, obs):
        label = jnp.logical_not(obs[self.n_inactive_input:]).astype(jnp.int32)
        label = label.reshape(1, )
        # return jnp.concatenate([label.reshape(1,), self.inactive_output])

        label = self.add_inactive_output(label)

        return label



