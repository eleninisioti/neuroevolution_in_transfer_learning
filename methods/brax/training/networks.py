# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Network definitions."""

import dataclasses
from typing import Any, Callable, Sequence, Tuple
import warnings

from brax.training import types
from brax.training.spectral_norm import SNDense
from flax import linen
import jax
import jax.numpy as jnp
import numpy as onp
ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


@dataclasses.dataclass
class FeedForwardNetwork:
  init: Callable[..., Any]
  apply: Callable[..., Any]



class MLPWithSkip(linen.Module):

    input_size: int
    output_size: int
    hidden_size: int
    mask:jnp.ndarray
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = False

    @linen.compact
    def __call__(self, data: jnp.ndarray,  Hs=None):
        #self.layer_sizes = [n_input+hidden_size+n_output+1]
        total_size = 1 + self.input_size + self.output_size + self.hidden_size
        if len(jnp.shape(data))>2:
            state = jnp.zeros((jnp.shape(data)[0],jnp.shape(data)[1], self.hidden_size))
            output = jnp.zeros((jnp.shape(data)[0], jnp.shape(data)[1],self.output_size))
            bias = jnp.ones((jnp.shape(data)[0],jnp.shape(data)[1],1))
        elif len(jnp.shape(data))>1:
            state = jnp.zeros((jnp.shape(data)[0], self.hidden_size))
            output = jnp.zeros((jnp.shape(data)[0], self.output_size))
            bias = jnp.ones((jnp.shape(data)[0],1))
            """
            mask = jax.tree_util.tree_map(lambda x: jnp.repeat(
              jnp.expand_dims(x, axis=0), jnp.shape(data)[0], axis=0),
          mask)
          """

        else:
            state = jnp.zeros((self.hidden_size, ))
            output = jnp.zeros((self.output_size, ))
            bias = jnp.ones((1,))
            #mask = mask

        activations = jnp.concatenate((bias, data, state, output), axis=len(jnp.shape(data))-1)

        dense_layer = linen.Dense(
            total_size,
            name=f'hidden',
            kernel_init=self.kernel_init,
            use_bias=self.bias)(
            activations)

        # Get the parameters (weights) of the dense layer
        weights = self.variables["params"]["hidden"]["kernel"]

        # Multiply weights with the mask
        masked_weights = weights * self.mask

        # Perform matrix multiplication with masked weights
        activations = jnp.dot(activations, masked_weights)



        for i in range(5):
            activations = activations
            activations = self.activation(activations)
        if len(jnp.shape(data))>2:
            action = activations[:,:,-self.output_size:]

        elif len(jnp.shape(data))>1:
            action = activations[:,-self.output_size:]
        else:
            action = activations[-self.output_size:]
        return action



class MLP(linen.Module):
  """MLP module."""
  layer_sizes: Sequence[int]
  activation: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  bias: bool = True
  layer_norm: bool = False

  @linen.compact
  def __call__(self, data: jnp.ndarray):
    hidden = data
    for i, hidden_size in enumerate(self.layer_sizes):
      hidden = linen.Dense(
          hidden_size,
          name=f'hidden_{i}',
          kernel_init=self.kernel_init,
          use_bias=self.bias)(
              hidden)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        hidden = self.activation(hidden)
        if self.layer_norm:
          hidden = linen.LayerNorm()(hidden)
    return hidden


class SNMLP(linen.Module):
  """MLP module with Spectral Normalization."""
  layer_sizes: Sequence[int]
  activation: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  bias: bool = True

  @linen.compact
  def __call__(self, data: jnp.ndarray):
    hidden = data
    for i, hidden_size in enumerate(self.layer_sizes):
      hidden = SNDense(
          hidden_size,
          name=f'hidden_{i}',
          kernel_init=self.kernel_init,
          use_bias=self.bias)(
              hidden)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        hidden = self.activation(hidden)
    return hidden


def make_policy_network(
    param_size: int,
    obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    layer_norm: bool = False) -> FeedForwardNetwork:
  """Creates a policy network."""
  policy_module = MLP(
      layer_sizes=list(hidden_layer_sizes) + [param_size],
      kernel_init=kernel_init,
      activation=activation,
      activate_final=True,
      layer_norm=layer_norm)

  def apply(processor_params, policy_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    return policy_module.apply(policy_params, obs)

  dummy_obs = jnp.zeros((1, obs_size))
  return FeedForwardNetwork(
      init=lambda key: policy_module.init(key, dummy_obs), apply=apply)


def make_value_network(
    obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu) -> FeedForwardNetwork:
  """Creates a policy network."""
  value_module = MLP(
      layer_sizes=list(hidden_layer_sizes) + [1],
      activation=activation,
      kernel_init=jax.nn.initializers.lecun_uniform())

  def apply(processor_params, policy_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    return jnp.squeeze(value_module.apply(policy_params, obs), axis=-1)

  dummy_obs = jnp.zeros((1, obs_size))
  return FeedForwardNetwork(
      init=lambda key: value_module.init(key, dummy_obs), apply=apply)

def make_policy_network_with_skip(
    param_size: int,
    obs_size: int,
        skip_connections_prob: float,
        mask: jnp.ndarray,
        preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    layer_norm: bool = False) -> FeedForwardNetwork:
  """Creates a policy network."""
  policy_module = MLPWithSkip(
     hidden_size=onp.sum(hidden_layer_sizes) ,
      activation=activation,
      kernel_init=kernel_init,
      mask=mask,
      input_size=obs_size,
      output_size=param_size)

  def apply(processor_params, policy_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    return policy_module.apply(policy_params, obs)

  dummy_obs = jnp.zeros((1, obs_size))
  return FeedForwardNetwork(
      init=lambda key: policy_module.init(key, dummy_obs), apply=apply)


def make_value_network_with_skip(
    obs_size: int,
        skip_connections_prob: float,
        mask: jnp.ndarray,

        preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu) -> FeedForwardNetwork:
  """Creates a policy network."""


  value_module = MLPWithSkip(
      hidden_size=onp.sum(hidden_layer_sizes) ,
      activation=activation,
      kernel_init=jax.nn.initializers.lecun_uniform(),
      mask=mask,
      input_size=obs_size,
      output_size=1)

  def apply(processor_params, policy_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    return jnp.squeeze(value_module.apply(policy_params, obs), axis=-1)

  dummy_obs = jnp.zeros((1, obs_size))
  return FeedForwardNetwork(
      init=lambda key: value_module.init(key, dummy_obs), apply=apply)



def make_q_network(
    obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    n_critics: int = 2) -> FeedForwardNetwork:
  """Creates a value network."""

  class QModule(linen.Module):
    """Q Module."""
    n_critics: int

    @linen.compact
    def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
      hidden = jnp.concatenate([obs, actions], axis=-1)
      res = []
      for _ in range(self.n_critics):
        q = MLP(
            layer_sizes=list(hidden_layer_sizes) + [1],
            activation=activation,
            kernel_init=jax.nn.initializers.lecun_uniform())(
                hidden)
        res.append(q)
      return jnp.concatenate(res, axis=-1)

  q_module = QModule(n_critics=n_critics)

  def apply(processor_params, q_params, obs, actions):
    obs = preprocess_observations_fn(obs, processor_params)
    return q_module.apply(q_params, obs, actions)

  dummy_obs = jnp.zeros((1, obs_size))
  dummy_action = jnp.zeros((1, action_size))
  return FeedForwardNetwork(
      init=lambda key: q_module.init(key, dummy_obs, dummy_action), apply=apply)


def make_model(
    layer_sizes: Sequence[int],
    obs_size: int,
    activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.swish,
    spectral_norm: bool = False,
) -> FeedForwardNetwork:
  """Creates a model.

  Args:
    layer_sizes: layers
    obs_size: size of an observation
    activation: activation
    spectral_norm: whether to use a spectral normalization (default: False).

  Returns:
    a model
  """
  warnings.warn(
      'make_model is deprecated, use make_{policy|q|value}_network instead.')
  dummy_obs = jnp.zeros((1, obs_size))
  if spectral_norm:
    module = SNMLP(layer_sizes=layer_sizes, activation=activation)
    model = FeedForwardNetwork(
        init=lambda rng1, rng2: module.init({
            'params': rng1,
            'sing_vec': rng2
        }, dummy_obs),
        apply=module.apply)
  else:
    module = MLP(layer_sizes=layer_sizes, activation=activation)
    model = FeedForwardNetwork(
        init=lambda rng: module.init(rng, dummy_obs), apply=module.apply)
  return model


def make_models(policy_params_size: int,
                obs_size: int) -> Tuple[FeedForwardNetwork, FeedForwardNetwork]:
  """Creates models for policy and value functions.

  Args:
    policy_params_size: number of params that a policy network should generate
    obs_size: size of an observation

  Returns:
    a model for policy and a model for value function
  """
  warnings.warn(
      'make_models is deprecated, use make_{policy|q|value}_network instead.')
  policy_model = make_model([32, 32, 32, 32, policy_params_size], obs_size)
  value_model = make_model([256, 256, 256, 256, 256, 1], obs_size)
  return policy_model, value_model
