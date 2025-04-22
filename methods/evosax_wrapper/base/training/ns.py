from src.training.base import BaseTrainer
from src.training.logging import Logger

from typing import NamedTuple, Optional, Tuple, Any, Callable, TypeAlias, Union, Dict
from jaxtyping import Float, PyTree

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import methods.evosax_wrapper.evosax as ex
from jax.experimental.shard_map import shard_map as shmap
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils

Params: TypeAlias = PyTree
Data: TypeAlias = PyTree
QDTask: TypeAlias = Callable[[Params, jax.Array, Optional[Data]], Tuple[Float, Float, Data]]

class TrainState(NamedTuple):
	archive: jax.Array
	es_state: jax.Array


class NoveltySearchTrainer(BaseTrainer):
	"""
	"""
	#-------------------------------------------------------------------
	strategy: ex.Strategy
	es_params: ex.EvoParams
	params_shaper: ex.ParameterReshaper
	task: QDTask
	fitness_shaper: ex.FitnessShaper
	n_devices: int
	#-------------------------------------------------------------------

	def __init__(
		self, 
		train_steps: int,
		strategy: Union[ex.Strategy, str],
		task: QDTask,
		params_shaper: ex.ParameterReshaper,
		popsize: Optional[int]=None,
		fitness_shaper: Optional[ex.FitnessShaper]=None,
		es_kws: Optional[Dict[str, Any]]={},
		es_params: Optional[ex.EvoParams]=None,
		logger: Optional[Logger]=None,
	    progress_bar: Optional[bool]=True,
	    n_devices: int=1):

		super().__init__(train_steps=train_steps, 
						 logger=logger, 
						 progress_bar=progress_bar)
		
		if isinstance(strategy, str):
			assert popsize is not None
			self.strategy = self.create_strategy(strategy, popsize, params_shaper.total_params, **es_kws) # type: ignore
		else:
			self.strategy = strategy

		if es_params is None:
			self.es_params = self.strategy.default_params
		else:
			self.es_params = es_params

		self.params_shaper = params_shaper

		self.task = task

		if fitness_shaper is None:
			self.fitness_shaper = ex.FitnessShaper()
		else:
			self.fitness_shaper = fitness_shaper

		self.n_devices = n_devices

	#-------------------------------------------------------------------

	def eval(self, *args, **kwargs):
		
		if self.n_devices == 1:
			fit, bd, eval_data = self._eval(*args, **kwargs)
		else:
			fit, bd, eval_data = self._eval_shmap(*args, **kwargs)

		return fit, bd, eval_data

	#-------------------------------------------------------------------

	def _eval(self, x: jax.Array, key: jax.Array, data: PyTree)->Tuple[Float, Float, Data]:
		
		params = self.params_shaper.reshape(x)
		_eval = jax.vmap(self.task, in_axes=(0, None, None))
		return _eval(params, key, data)

	#-------------------------------------------------------------------

	def _eval_shmap(self, x: jax.Array, key: jax.Array, data: PyTree)->Tuple[Float, Float, Data]:
		
		devices = mesh_utils.create_device_mesh((self.n_devices,))
		device_mesh = Mesh(devices, axis_names=("p"))

		_eval = lambda x, k: self.task(self.params_shaper.reshape_single(x), k, data)
		batch_eval = jax.vmap(_eval, in_axes=(0,None))
		sheval = shmap(batch_eval, 
					   mesh=device_mesh, 
					   in_specs=(P("p",), P()),
					   out_specs=(P("p"), P("p")),
					   check_rep=False)

		return sheval(x, key)

	#-------------------------------------------------------------------

	def train_step(self, state: TrainState, key: jr.PRNGKeyArray, data: Optional[Data] = None) -> Tuple[TrainState, Any]:
		ask_key, eval_key = jr.split(key, 2)
		x, es_state = self.strategy.ask(ask_key, state.es_state, self.es_params)
		fitness, descriptors, eval_data = self.eval(x, eval_key, data)
		f = self.fitness_shaper.apply(x, fitness)
		es_state = self.strategy.tell(x, f, es_state, self.es_params)
		archive = self.update_archive(state.archive, descriptors)
		return TrainState(archive=archive, es_state=es_state), {"fitness": fitness, "data": eval_data}

	#-------------------------------------------------------------------

	def update_archive(self, archive: jax.Array, descriptors: jax.Array)->jax.Array:
		pass


