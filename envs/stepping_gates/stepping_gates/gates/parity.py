import jax.numpy as jnp
from stepping_gates.gates.gate import Gate
from stepping_gates.gates.increment import Incr

class Parity(Gate):

    def __init__(self, n_input_circuit, n_output_circuit):


        self.n_input = n_input_circuit
        self.n_output = n_output_circuit

    def reset(self, key):
        return jnp.array([0]*self.n_input)

    def function(self, obs):
        return (jnp.sum(obs) % 2)


