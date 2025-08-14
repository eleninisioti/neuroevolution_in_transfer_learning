
from typing import NamedTuple, Optional, Union
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
from typing import Mapping, NamedTuple, Tuple
import equinox.nn as nn
from flax import linen
import pickle


from typing import NamedTuple

class PolicyState(NamedTuple):
    weights: jax.Array
    adj: jax.Array
    rnn_state: Optional[jax.Array]


class MLP(eqx.Module):
    action_dims: int
    obs_dims: int
    mlp: Optional[jax.Array]
    max_nodes: int

    #norm_data: Optional[jax.Array]
    #pretrained_params: Optional[jax.Array]

    def __init__(self,  action_dims, obs_dims,*, key: jax.Array, max_nodes):

        self.action_dims = action_dims
        self.obs_dims = obs_dims
        self.max_nodes = max_nodes
        self.mlp = nn.MLP(obs_dims,
                          action_dims,
                        16, 2,
                          activation=linen.relu, final_activation=lambda x: x,
                      key=key, use_bias=True, use_final_bias=True)



    def initialize(self, init_key, current_gen=None):

        # extract weights and biases

        final_policy = PolicyState(weights=jnp.zeros((1,1)), adj=jnp.zeros((1,1)), rnn_state=jnp.zeros((jnp.zeros((1,1)).shape[0],)))
        interm_policies =jax.tree_map(lambda x: x[None,:], final_policy)
        return final_policy, interm_policies



    def get_phenotype(self, mlp):

        # extract weights and biases
        params = mlp.layers

        width = 0
        height = 0
        for el in params:
            kernel = el.weight
            height += kernel.shape[1]
            width += kernel.shape[0]

        weights = jnp.zeros((width, height))
        start_x = 0
        start_y = self.obs_dims
        for el in params:
            kernel = el.weight
            weights = weights.at[start_x:start_x + kernel.shape[1], start_y:start_y + kernel.shape[0]].set(jnp.transpose(el.weight))
            start_x += kernel.shape[1]
            start_y += kernel.shape[0]
        # process bias
        width = 0
        for el in params:
            kernel = el.bias
            width += kernel.shape[0]
        bias = jnp.zeros((width,))
        start_x = 0
        for el in params:
            bias = bias.at[start_x:start_x + el.bias.shape[0]].set(el.bias)
            start_x += el.bias.shape[0]
        adj = jnp.where(weights, 1.0, 0.0)
        final_policy = PolicyState(weights=weights, adj=adj, rnn_state=jnp.zeros((weights.shape[0],)))
        interm_policies =jax.tree_map(lambda x: x[None,:], final_policy)
        return final_policy, interm_policies

    def __call__(self, obs: jax.Array, state: PolicyState, key: jax.Array, obs_size=None, action_size=None) -> Tuple[jax.Array, PolicyState]:
        #jax.debug.print("obs: {}",obs)
        a = self.mlp(obs)
        #jax.debug.print("inside model: {}", a)

        return a, state


class RNN(eqx.Module):
    action_dims: int
    obs_dims: int
    weights: Optional[jax.Array]
    policy_iters: int
    #norm_data: Optional[jax.Array]
    #pretrained_params: Optional[jax.Array]

    def __init__(self,  action_dims, obs_dims, total_nodes, *, key: jax.Array ):

        self.action_dims = action_dims
        self.obs_dims = obs_dims
        self.policy_iters = 5
        self.weights = jr.normal(key, (total_nodes,
                                       total_nodes))  # self.mlp = nn.MLP(8, action_dims, 32, 4, activation=linen.swish, final_activation=lambda x: x,

    def initialize(self, init_key):

        # extract weights and biases
        weights = self.weights
        adj = jnp.where(weights, 1, 0)
        final_policy = PolicyState(weights=weights,  adj=adj, rnn_state=jnp.zeros((weights.shape[0],)))
        interm_policies =jax.tree_map(lambda x: x[None,:], final_policy)
        return final_policy, interm_policies



    def get_phenotype(self, weights):

        # extract weights and biases
        adj = jnp.where(weights, 1.0, 0.0)
        final_policy = PolicyState(weights=weights,  adj=adj, rnn_state=jnp.zeros((weights.shape[0],)))
        interm_policies =jax.tree_map(lambda x: x[None,:], final_policy)
        return final_policy, interm_policies

    def __call__(self, obs: jax.Array, state: PolicyState, key: jax.Array, obs_size=None, action_size=None) -> Tuple[jax.Array, PolicyState]:
        w = (state.weights)

        def set_input(h):
            h = h.at[0].set(1)
            h = h.at[1:self.obs_dims + 1].set(obs)
            return h

        def rnn_step(h):
            h = set_input(h)
            h = linen.tanh(jnp.matmul(w, h))
            return h

        h = jax.lax.fori_loop(0, self.policy_iters, lambda _, h: rnn_step(h), state.rnn_state)

        a = h[-self.action_dims:]
        state = state._replace(rnn_state=h)
        return a, state


class CNN(eqx.Module):
    """CNN module with three heads for MinAtar games."""
    
    action_dims: int
    obs_dims: int
    max_nodes: int
    conv: nn.Conv2d
    fc_hidden: nn.Linear
    output: nn.Linear
    activation: callable

    def __init__(self, action_dims, obs_dims, total_nodes, *, key: jax.Array):
        self.action_dims = action_dims
        self.obs_dims = obs_dims
        self.max_nodes = total_nodes
        
        # Split key for different layers
        key, conv_key, fc_key, out1_key, out2_key, out3_key = jr.split(key, 6)
        
        # Single conv layer: 7 input channels -> 32 output channels, 3x3 kernel, stride=1
        self.conv = nn.Conv2d(7, 32, kernel_size=3, stride=1, key=conv_key)
        
        # Calculate linear units after conv layer
        # For 10x10 input with 3x3 kernel and stride=1: (10-3)/1 + 1 = 8
        # So 8x8x32 = 2048 features
        num_linear_units = 8 * 8 * 32
        
        # Hidden fully connected layer
        self.fc_hidden = nn.Linear(num_linear_units, 256, key=fc_key)
        
        # Single output head for action_dims actions
        self.output = nn.Linear(256, action_dims, key=out1_key)
        
        self.activation = jnn.relu

    def initialize(self, init_key):
        # For CNN, we need to create a compatible policy state
        # The actual CNN parameters are in the module itself
        # Use a fixed size for the evolutionary framework compatibility
        weights = jnp.zeros((100, 100))  # Fixed size for compatibility
        adj = jnp.zeros((100, 100))
        final_policy = PolicyState(weights=weights, adj=adj, rnn_state=jnp.zeros((100,)))
        interm_policies = jax.tree_map(lambda x: x[None, :], final_policy)
        return final_policy, interm_policies

    def get_phenotype(self, weights):
        # CNN has fixed architecture, so phenotype is the same
        # Use a fixed size for the evolutionary framework compatibility
        adj = jnp.zeros((100, 100))
        final_policy = PolicyState(weights=weights, adj=adj, rnn_state=jnp.zeros((100,)))
        interm_policies = jax.tree_map(lambda x: x[None, :], final_policy)
        return final_policy, interm_policies

    def __call__(self, obs: jax.Array, state: PolicyState, key: jax.Array, obs_size=None, action_size=None) -> Tuple[jax.Array, PolicyState]:
        # Handle MinAtar observations: reshape from (10, 10, 7) to (7, 10, 10) for Conv2d
        if obs.shape == (10, 10, 7):  # MinAtar format
            x = jnp.transpose(obs, (2, 0, 1))  # Transpose to (7, 10, 10) for Conv2d
        else:
            x = obs
        
        # Apply conv layer
        x = self.activation(self.conv(x))
        
        # Flatten and apply hidden FC layer
        x = x.reshape(-1)  # Flatten to 1D
        x = self.activation(self.fc_hidden(x))
        
        # Get single output
        output = self.output(x)
        
        return output, state


def make_model(config, key):
    """ Creates a direct encoding
    """

    key, key_model = jr.split(key)

    action_size = config["env_config"]["action_size"]
    input_size = config["env_config"]["observation_size"]
    max_nodes = action_size + input_size + config["model_config"]["model_params"]["max_hidden_neurons"] + 1

    if config["model_config"]["network_type"] == "MLP":

        model = MLP( key=key_model,
                     obs_dims=input_size,
                     action_dims=action_size,
                     max_nodes=max_nodes
                    )
    elif config["model_config"]["network_type"] == "RNN":

        model = RNN(key=key_model,
                    obs_dims=input_size,
                    action_dims=action_size,
                    total_nodes=max_nodes,
                    )
    elif config["model_config"]["network_type"] == "CNN":

        model = CNN(key=key_model,
                    obs_dims=input_size,
                    action_dims=action_size,
                    total_nodes=max_nodes,
                    )

    return model
