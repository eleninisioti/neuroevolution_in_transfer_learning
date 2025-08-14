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

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import time
from typing import Callable, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from methods.brax_wrapper import acting
from brax.training import gradients
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision

from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.v1 import envs as envs_v1
from etils import epath
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from orbax import checkpoint as ocp
from methods.brax_wrapper.wrappers.training import wrap as brax_wrap
from methods.brax_wrapper.wrappers.training_gymnax_vision import wrap as gymnax_wrap
#from brax.envs.wrappers.training import wrap as brax_wrap
import gymnax
from envs.stepping_gates.stepping_gates.envs.wrappers import wrap as dgates_wrap
import gymnasium
import numpy as onp
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""
  optimizer_state: optax.OptState
  params: ppo_losses.PPONetworkParams
  normalizer_params: running_statistics.RunningStatisticsState
  env_steps: jnp.ndarray
  env_params: dict


def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
  # brax user code is sometimes ambiguous about weak_type.  in order to
  # avoid extra jit recompilations we strip all weak types from user input
  def f(leaf):
    leaf = jnp.asarray(leaf)
    return leaf.astype(leaf.dtype)
  return jax.tree_util.tree_map(f, tree)


def train(
    environment: Union[envs_v1.Env, envs.Env],
    num_timesteps: int,
    episode_length: int,
    save_params_fn,
    gymnax_env_params, # this is needed for gymnax
    skip_connections_prob: float=0.0,
    num_neurons: int=32, # number of neurons used in each layer of the policy network. value network will be this times 8
    num_layers: int=4, # number of layers used in policy network. value network will be this +1
    wrap_env: bool = True,
    action_repeat: int = 1,
    num_envs: int = 1,
    max_devices_per_host: Optional[int] = None,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    seed: int = 0,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_evals: int = 1,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    deterministic_eval: bool = True,
    network_factory: types.NetworkFactory[
        ppo_networks_vision.PPONetworks
    ] = ppo_networks_vision.make_ppo_networks_vision,
    network_factory_with_skip: types.NetworkFactory[
        ppo_networks_vision.PPONetworks
    ] = ppo_networks_vision.make_ppo_networks_vision,

    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    normalize_advantage: bool = True,
    eval_env: Optional[envs.Env] = None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    restore_checkpoint_path: Optional[str] = None,

):
  """PPO training.

  Args:
    environment: the environment to train
    num_timesteps: the total number of environment steps to use during training
    episode_length: the length of an environment episode
    wrap_env: If True, wrap the environment for training. Otherwise use the
      environment as is.
    action_repeat: the number of timesteps to repeat an action
    num_envs: the number of parallel environments to use for rollouts
      NOTE: `num_envs` must be divisible by the total number of chips since each
        chip gets `num_envs // total_number_of_chips` environments to roll out
      NOTE: `batch_size * num_minibatches` must be divisible by `num_envs` since
        data generated by `num_envs` parallel envs gets used for gradient
        updates over `num_minibatches` of data, where each minibatch has a
        leading dimension of `batch_size`
    max_devices_per_host: maximum number of chips to use per host process
    num_eval_envs: the number of envs to use for evluation. Each env will run 1
      episode, and all envs run in parallel during eval.
    learning_rate: learning rate for ppo loss
    entropy_cost: entropy reward for ppo loss, higher values increase entropy of
      the policy
    discounting: discounting rate
    seed: random seed
    unroll_length: the number of timesteps to unroll in each environment. The
      PPO loss is computed over `unroll_length` timesteps
    batch_size: the batch size for each minibatch SGD step
    num_minibatches: the number of times to run the SGD step, each with a
      different minibatch with leading dimension of `batch_size`
    num_updates_per_batch: the number of times to run the gradient update over
      all minibatches before doing a new environment rollout
    num_evals: the number of evals to run during the entire training run.
      Increasing the number of evals increases total training time
    num_resets_per_eval: the number of environment resets to run between each
      eval. The environment resets occur on the host
    normalize_observations: whether to normalize observations
    reward_scaling: float scaling for reward
    clipping_epsilon: clipping epsilon for PPO loss
    gae_lambda: General advantage estimation lambda
    deterministic_eval: whether to run the eval with a deterministic policy
    network_factory: function that generates networks for policy and value
      functions
    progress_fn: a user-defined callback function for reporting/plotting metrics
    normalize_advantage: whether to normalize advantage estimate
    eval_env: an optional environment for eval only, defaults to `environment`
    policy_params_fn: a user-defined callback function that can be used for
      saving policy checkpoints
    randomization_fn: a user-defined callback function that generates randomized
      environments
    restore_checkpoint_path: the path used to restore previous model params

  Returns:
    Tuple of (make_policy function, network params, metrics)
  """
  assert batch_size * num_minibatches % num_envs == 0
  xt = time.time()

  process_count = jax.process_count()
  process_id = jax.process_index()
  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count
  if max_devices_per_host:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  logging.info(
      'Device count: %d, process count: %d (id %d), local device count: %d, '
      'devices to be used count: %d', jax.device_count(), process_count,
      process_id, local_device_count, local_devices_to_use)
  device_count = local_devices_to_use * process_count

  # The number of environment steps executed for every training step.
  env_step_per_training_step = (
      batch_size * unroll_length * num_minibatches * action_repeat)
  num_evals_after_init = max(num_evals - 1, 1)
  # The number of training_step calls per training_epoch call.
  # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step *
  #                                 num_resets_per_eval))
  num_training_steps_per_epoch = np.ceil(
      num_timesteps
      / (
          num_evals_after_init
          * env_step_per_training_step
          * max(num_resets_per_eval, 1)
      )
  ).astype(int)

  key = jax.random.PRNGKey(seed)
  global_key, local_key = jax.random.split(key)
  del key
  local_key = jax.random.fold_in(local_key, process_id)
  local_key, key_env, eval_key = jax.random.split(local_key, 3)
  # key_networks should be global, so that networks are initialized the same
  # way for different processes.
  key_policy, key_value = jax.random.split(global_key)
  del global_key

  assert num_envs % device_count == 0

  env = environment
  gymnax_env = True
  if wrap_env:
    v_randomization_fn = None
    if randomization_fn is not None:
      randomization_batch_size = num_envs // local_device_count
      # all devices gets the same randomization rng
      randomization_rng = jax.random.split(key_env, randomization_batch_size)
      v_randomization_fn = functools.partial(
          randomization_fn, rng=randomization_rng
      )
    if isinstance(environment, envs.Env):
      env = brax_wrap(
          environment,
          episode_length=episode_length,
          action_repeat=action_repeat,
          randomization_fn=v_randomization_fn,
      )
    elif gymnax_env==True:
        env = gymnax_wrap(
            environment,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=v_randomization_fn,
        )
    else:

      env = dgates_wrap(
          environment,
          episode_length=episode_length,
          action_repeat=action_repeat,
          randomization_fn=v_randomization_fn,
      )
  key_envs = jax.random.split(key_env, num_envs)
  key_envs = jnp.reshape(key_envs,
                         (local_devices_to_use, -1) + key_envs.shape[1:])

  init_env_params = jnp.zeros((1,)).astype(jnp.int32)
  
  if isinstance(environment, envs.Env):
    reset_fn = jax.jit(jax.vmap(env.reset, in_axes=(0)))

    env_state = reset_fn(key_envs)
  else:
    reset_fn = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))

    env_state = reset_fn(key_envs, init_env_params)

  normalize = lambda x, y: x
  if normalize_observations:
    normalize = running_statistics.normalize

  def make_mask_for_mlpwithskip(arch):
    total_nodes = arch["input"]+ onp.sum(arch["hidden"]) + arch["output"] + 1 #bias
    mask = jnp.zeros((total_nodes, total_nodes))

    mask =mask.at[0,...].set(1) # bias
    arch_hidden = (arch["input"],) + arch["hidden"]
    input_layer_start = 1

    for idx, hidden_layer in enumerate(arch_hidden[:-1]):


        output_layer_start = input_layer_start+hidden_layer
        output_layer_end = output_layer_start+arch_hidden[idx+1]

        #print(input_layer_start,input_layer_start+hidden_layer)
        #print(output_layer_start,output_layer_end)
        mask = mask.at[input_layer_start:input_layer_start+hidden_layer,output_layer_start:output_layer_end].set(1)
        input_layer_start = output_layer_start

    #print(input_layer_start, input_layer_start+arch_hidden[-1])
    mask = mask.at[input_layer_start:input_layer_start+arch_hidden[-1],input_layer_start+arch_hidden[-1]:input_layer_start+arch_hidden[-1]+arch["output"]].set(1)

    size = mask.shape[0]

    N = int(skip_connections_prob*(size*size/2))
    # Get the upper triangle indices (excluding the diagonal)
    upper_triangle_indices = jnp.triu_indices(size, k=1)

    # Get the total number of upper triangle elements
    num_upper_triangle_elements = upper_triangle_indices[0].shape[0]

    # Ensure N does not exceed the number of available elements
    N = min(N, num_upper_triangle_elements)

    # Generate a random key for JAX
    key = jax.random.PRNGKey(0)

    # Randomly select N unique indices from the upper triangle indices
    random_indices = jax.random.choice(key, num_upper_triangle_elements, shape=(N,), replace=False)

    # Set the selected upper triangle elements to ones
    mask = mask.at[upper_triangle_indices[0][random_indices], upper_triangle_indices[1][random_indices]].set(1)

    return mask


  if skip_connections_prob:
      arch = {"input": env_state.obs.shape[-1],
              "hidden": (num_neurons,) * num_layers,
              "output": env.action_size*2}
      policy_mask = make_mask_for_mlpwithskip(arch)
      arch = {"input": env_state.obs.shape[-1],
              "hidden": (num_neurons*4,)*(num_layers+1),
              "output":1}
      value_mask = make_mask_for_mlpwithskip(arch)

      ppo_network = network_factory_with_skip(
          env_state.obs.shape[-1],
          env.action_size,
          mask_policy=policy_mask,
          mask_value=value_mask,
          policy_hidden_layer_sizes= (num_neurons,) * num_layers,
        value_hidden_layer_sizes=(num_neurons*2,)*(num_layers+1),
          preprocess_observations_fn=normalize,
      skip_connections_prob=skip_connections_prob)
  else:

      if isinstance(environment, envs.Env):
          action_size = env.action_size
      else:
          action_size = env.num_actions

      #obs_shape = jax.tree_util.tree_map(
      #    lambda x: x.shape[1:], env_state.obs
      #)  # Discard batch axis over envs.
      obs_shape = {"pixels/": env_state.obs["pixels/"].shape[2:]} 
      ppo_network = network_factory(
          obs_shape,
          action_size,
          policy_hidden_layer_sizes= (num_neurons,) * num_layers,
         value_hidden_layer_sizes=(num_neurons*8,)*(num_layers+1),
          preprocess_observations_fn=normalize)

  make_policy = ppo_networks.make_inference_fn(ppo_network)

  optimizer = optax.adam(learning_rate=learning_rate)

  loss_fn = functools.partial(
      ppo_losses.compute_ppo_loss,
      ppo_network=ppo_network,
      entropy_cost=entropy_cost,
      discounting=discounting,
      reward_scaling=reward_scaling,
      gae_lambda=gae_lambda,
      clipping_epsilon=clipping_epsilon,
      normalize_advantage=normalize_advantage)

  gradient_update_fn = gradients.gradient_update_fn(
      loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)

  def minibatch_step(
      carry, data: types.Transition,
      normalizer_params: running_statistics.RunningStatisticsState):
    optimizer_state, params, key = carry
    key, key_loss = jax.random.split(key)
    (_, metrics), params, optimizer_state = gradient_update_fn(
        params,
        normalizer_params,
        data,
        key_loss,
        optimizer_state=optimizer_state)

    return (optimizer_state, params, key), metrics

  def sgd_step(carry, unused_t, data: types.Transition,
               normalizer_params: running_statistics.RunningStatisticsState):
    optimizer_state, params, key = carry
    key, key_perm, key_grad = jax.random.split(key, 3)

    def convert_data(x: jnp.ndarray):
      x = jax.random.permutation(key_perm, x)
      x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
      return x

    shuffled_data = jax.tree_util.tree_map(convert_data, data)
    (optimizer_state, params, _), metrics = jax.lax.scan(
        functools.partial(minibatch_step, normalizer_params=normalizer_params),
        (optimizer_state, params, key_grad),
        shuffled_data,
        length=num_minibatches)
    return (optimizer_state, params, key), metrics

  def training_step(
      carry: Tuple[TrainingState, envs.State, PRNGKey],
      unused_t) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
    training_state, state, key = carry
    key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

    policy = make_policy(
        (training_state.normalizer_params, training_state.params.policy))

    def f(carry, unused_t):
      current_state, current_key = carry
      current_key, next_key = jax.random.split(current_key)
      next_state, data = acting.generate_unroll(
          env,
          current_state,
          policy,
          current_key,
          unroll_length=unroll_length,
          env_params=gymnax_env_params,
          extra_fields=('truncation',)) # this is used in brax

      return (next_state, next_key), data

    (state, _), data = jax.lax.scan(
        f, (state, key_generate_unroll), (),
        length=batch_size * num_minibatches // num_envs)
    # Have leading dimensions (batch_size * num_minibatches, unroll_length)
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
    data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
                                  data)
    assert data.discount.shape[1:] == (unroll_length,)

    # Update normalization params and normalize observations.
    normalizer_params = running_statistics.update(
        training_state.normalizer_params,
        data.observation["pixels/"],
        pmap_axis_name=_PMAP_AXIS_NAME)

    (optimizer_state, params, _), metrics = jax.lax.scan(
        functools.partial(
            sgd_step, data=data, normalizer_params=normalizer_params),
        (training_state.optimizer_state, training_state.params, key_sgd), (),
        length=num_updates_per_batch)

    new_training_state = TrainingState(
        optimizer_state=optimizer_state,
        params=params,
        normalizer_params=normalizer_params,
        env_params=training_state.env_params,
        env_steps=training_state.env_steps + env_step_per_training_step)
    return (new_training_state, state, new_key), metrics

  def training_epoch(training_state: TrainingState, state: envs.State,
                     key: PRNGKey) -> Tuple[TrainingState, envs.State, Metrics]:
    (training_state, state, _), loss_metrics = jax.lax.scan(
        training_step, (training_state, state, key), (),
        length=num_training_steps_per_epoch)
    loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
    return training_state, state, loss_metrics

  training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

  # Note that this is NOT a pure jittable method.
  def training_epoch_with_timing(
      training_state: TrainingState, env_state: envs.State,
      key: PRNGKey) -> Tuple[TrainingState, envs.State, Metrics]:
    nonlocal training_walltime
    t = time.time()
    training_state, env_state = _strip_weak_type((training_state, env_state))
    result = training_epoch(training_state, env_state, key)
    training_state, env_state, metrics = _strip_weak_type(result)

    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (num_training_steps_per_epoch *
           env_step_per_training_step *
           max(num_resets_per_eval, 1)) / epoch_training_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in metrics.items()}
    }
    return training_state, env_state, metrics  # pytype: disable=bad-return-type  # py311-upgrade

  # Initialize model params and training state.
  init_params = ppo_losses.PPONetworkParams(
      policy=ppo_network.policy_network.init(key_policy),
      value=ppo_network.value_network.init(key_value),
  )

  training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
      optimizer_state=optimizer.init(init_params),  # pytype: disable=wrong-arg-types  # numpy-scalars
      params=init_params,
      env_params=init_env_params,
      normalizer_params=running_statistics.init_state(
          specs.Array(env_state.obs["pixels/"].shape[-1:], jnp.dtype('float32'))),
      env_steps=0)

  if (
      restore_checkpoint_path is not None
      and epath.Path(restore_checkpoint_path).exists()
  ):
    logging.info('restoring from checkpoint %s', restore_checkpoint_path)
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    target = training_state.normalizer_params, init_params
    (normalizer_params, init_params) = orbax_checkpointer.restore(
        restore_checkpoint_path, item=target
    )
    training_state = training_state.replace(
        normalizer_params=normalizer_params, params=init_params
    )

  if num_timesteps == 0:
    return (
        make_policy,
        (training_state.normalizer_params, training_state.params),
        {},
    )

  training_state = jax.device_put_replicated(
      training_state,
      jax.local_devices()[:local_devices_to_use])

  if not eval_env:
    eval_env = environment
  if wrap_env:
    if randomization_fn is not None:
      v_randomization_fn = functools.partial(
          randomization_fn, rng=jax.random.split(eval_key, num_eval_envs)
      )
    if isinstance(environment, envs.Env):
        eval_env = brax_wrap(
            eval_env,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=v_randomization_fn,
        )
    elif gymnax_env==True:
        eval_env = gymnax_wrap(
            environment,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=v_randomization_fn,
        )
    else:

      eval_env = dgates_wrap(
          environment,
          episode_length=episode_length,
          action_repeat=action_repeat,
          randomization_fn=v_randomization_fn,
      )

  evaluator = acting.Evaluator(
      eval_env,
      functools.partial(make_policy, deterministic=deterministic_eval),
      num_eval_envs=num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=eval_key)

  # Run initial eval
  metrics = {}
  if process_id == 0 and num_evals > 1:
    metrics = evaluator.run_evaluation(
        _unpmap(
            (training_state.normalizer_params, training_state.params.policy)),
        training_metrics={},
    env_params=gymnax_env_params)
    logging.info(metrics)
    progress_fn((0, training_state.env_params, metrics))

  training_metrics = {}
  training_walltime = 0
  current_step = 0
  for it in range(num_evals_after_init):
    logging.info('starting iteration %s %s', it, time.time() - xt)

    for _ in range(max(num_resets_per_eval, 1)):
      # optimization
      epoch_key, local_key = jax.random.split(local_key)
      epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
      (training_state, env_state, training_metrics) = (
          training_epoch_with_timing(training_state, env_state, epoch_keys)
      )
      current_step = int(_unpmap(training_state.env_steps))

      key_envs = jax.vmap(
          lambda x, s: jax.random.split(x[0], s),
          in_axes=(0, None))(key_envs, key_envs.shape[1])
      # TODO: move extra reset logic to the AutoResetWrapper.
      env_state = reset_fn(key_envs) if num_resets_per_eval > 0 else env_state

    if process_id == 0:
      # Run evals.
      metrics = evaluator.run_evaluation(
          _unpmap(
              (training_state.normalizer_params, training_state.params.policy)),
          training_metrics,
      gymnax_env_params)


      def change_task(env_params):
          new_task = jnp.minimum(env_params[0][0] + 1, env.num_tasks).astype(jnp.int32)
          return jnp.array([[new_task]])


      new_env_params = jax.lax.cond(metrics["eval/episode_reward"] >= env.reward_for_solved,
                                    lambda x: change_task(x), lambda x: x,
                                    training_state.env_params)
      

      if metrics["eval/episode_reward"] >= env.reward_for_solved:
          save_params_fn( training_state)

      training_state = TrainingState(
          optimizer_state=training_state.optimizer_state,
          params=training_state.params,
          normalizer_params=training_state.normalizer_params,
          env_params=new_env_params,
          env_steps=training_state.env_steps )
      logging.info(metrics)
      progress_fn((current_step, training_state.env_params, metrics))
      params = _unpmap(
          (training_state.normalizer_params, training_state.params)
      )
      policy_params_fn(current_step, make_policy, params)

  total_steps = current_step
  assert total_steps >= num_timesteps

  # If there was no mistakes the training_state should still be identical on all
  # devices.
  pmap.assert_is_replicated(training_state)
  params = _unpmap(
      (training_state.normalizer_params, training_state.params.policy))
  logging.info('total steps: %s', total_steps)
  pmap.synchronize_hosts()
  return (make_policy, params, metrics, training_state)