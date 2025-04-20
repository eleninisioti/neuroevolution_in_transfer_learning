import jax.numpy as jnp
from stepping_gates.gates.gate import Gate

class XOR(Gate):

    def __init__(self, n_input_circuit, n_output_circuit,n_control_bits):


        self.episode_length = 4

        self.n_input = 2
        self.n_output = 1

        super().set_inactive_input(n_input_circuit)
        super().__init__(n_control_bits, n_output_circuit)

        self.n_inactive_output = n_output_circuit - self.n_output
        self.inactive_output = jnp.array([2] * self.n_inactive_output,
                                        dtype=jnp.int32)  # we need to set the bits that the gate is not using


    def reset(self, key):
        return jnp.concatenate([self.inactive_input, jnp.array([0, 0]) ])



    def function(self, obs):

        o1 = jnp.logical_not(obs[self.n_inactive_input])
        o2 = jnp.logical_not(obs[self.n_inactive_input+1])
        o3 = jnp.logical_not(jnp.logical_and(o1, obs[self.n_inactive_input+1]))
        o4 = jnp.logical_not(jnp.logical_and(obs[self.n_inactive_input], o2))
        label = jnp.logical_not(jnp.logical_and(o3, o4)).astype(jnp.int32)
        #return jnp.concatenate([label.reshape(1,), self.inactive_output])
        label = label.reshape(1, )

        label = self.add_inactive_output(label)

        return label


