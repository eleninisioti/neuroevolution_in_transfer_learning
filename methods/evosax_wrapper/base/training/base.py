

from methods.evosax_wrapper.base.training.utils import progress_bar_scan, progress_bar_fori
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import Callable, Optional, Tuple, Any, TypeAlias, Union
import jax.experimental.host_callback as hcb

from jaxtyping import PyTree

Data: TypeAlias = PyTree[...]
TaskParams: TypeAlias = PyTree[...]
TrainState: TypeAlias = PyTree[...]


from methods.evosax_wrapper.base.training.logging import Logger

class BaseTrainer(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	train_steps: int
	logger: Optional[Logger]
	progress_bar: Optional[bool]
	#-------------------------------------------------------------------

	def __init__(self, 
				 train_steps: int, 
				 logger: Optional[Logger]=None,
				 progress_bar: Optional[bool]=False):
		
		self.train_steps = train_steps
		self.progress_bar = progress_bar
		self.logger = logger

	#-------------------------------------------------------------------

	def __call__(self, key: jr.PRNGKey):

		return self.init_and_train(key)

	#-------------------------------------------------------------------

	def train(self, state: TrainState, key: jax.Array, data: Optional[Data]=None)->Tuple[TrainState, Data]:

		def _step(c, x):
			s, k = c
			k, k_ = jr.split(k)
			s, data = self.train_step(s, k_)
			
			if self.logger is not None:
				self.logger.log(s, data)

			return [s, k], {"states": s, "metrics": data}

		if self.progress_bar:
			_step = progress_bar_scan(self.train_steps)(_step) #type: ignore

		[state, key], data = jax.lax.scan(_step, [state, key], jnp.arange(self.train_steps))

		return state, data

	#-------------------------------------------------------------------

	def train_(self, state: TrainState, key: jax.Array, data: Optional[Data]=None)->TrainState:

		def _step(i, c):
			s, k, task_params = c
			k, k_ = jr.split(k)
			#if self.logger is not None:
			#	dummy_data = {"fitness": jnp.array([0]), "interm_policies": [], "best_indiv": []}
			s, data, task_params = self.train_step(s, k_, task_params, i)

			def save_params(data):
				for dev_step in range(self.logger.dev_steps + 1):
					current_dev = jax.tree_map(lambda x: x[data["best_indiv"], 0, dev_step, ...],
											   data["interm_policies"])
					self.logger.save_chkpt(current_dev, task_params, jnp.array(dev_step))

			jax.lax.cond(state.best_fitness >= self.reward_for_solved,
											   lambda x: save_params(x), lambda x: None, data)



			self.logger.log(s, data, task_params)

			return [s, k, task_params]

		if self.progress_bar:
			_step = progress_bar_fori(self.train_steps)(_step) #type: ignore

		task_params_init = 0

		[state, key, task_params] = jax.lax.fori_loop(0, self.train_steps, _step, [state, key, task_params_init])
		return state

	#-------------------------------------------------------------------

	def log(self, data):
		hcb.id_tap(
			lambda d, *_: wandb.log(d), data
		)

	#-------------------------------------------------------------------

	def init_and_train(self, key: jr.PRNGKey, data: Optional[Data]=None)->Tuple[TrainState, Data]:
		init_key, train_key = jr.split(key)
		state = self.initialize(init_key)
		return self.train(state, train_key, data)

	#-------------------------------------------------------------------

	def init_and_train_(self, key: jr.PRNGKey, data: Optional[Data]=None)->TrainState:
		init_key, train_key = jr.split(key)
		state = self.initialize(init_key)
		return self.train_(state, train_key, data)

	#-------------------------------------------------------------------

	def train_step(self, state: TrainState, key: jr.PRNGKey, data: Optional[Data]=None)->Tuple[TrainState, Any]:
		raise NotImplementedError

	#-------------------------------------------------------------------

	def initialize(self, key: jr.PRNGKey)->TrainState:
		raise NotImplementedError

	#-------------------------------------------------------------------

	def train_from_model_ckpt(self, ckpt_path: str, key: Optional[jax.Array]=None)->Tuple[TrainState, Data]:
		raise NotImplementedError

	#-------------------------------------------------------------------

	def train_from_model_ckpt_(self, ckpt_path: str, key: Optional[jax.Array]=None)->Tuple[TrainState, Data]:
		raise NotImplementedError

	#-------------------------------------------------------------------

	def train_from_training_ckpt(self, ckpt_path: str, key: Optional[jax.Array]=None)->Tuple[TrainState, Data]:
		raise NotImplementedError

	#-------------------------------------------------------------------

	def train_from_training_ckpt_(self, ckpt_path: str, key: Optional[jax.Array]=None)->Tuple[TrainState, Data]:
		raise NotImplementedError




class BaseMultiTrainer(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	trainers: list[BaseTrainer]
	transform_fns: Union[list[Callable[[TrainState, TrainState], TrainState]], 
						 Callable[[TrainState, TrainState], TrainState]]
	#-------------------------------------------------------------------

	def __init__(self, 
				 trainers: list[BaseTrainer], 
				 transform_fns: Union[list[Callable[[TrainState, TrainState], TrainState]], 
				 					  Callable[[TrainState, TrainState], TrainState]]):
		
		self.trainers = trainers
		self.transform_fns = transform_fns

	#-------------------------------------------------------------------

	def init_and_train_(self, key: jax.Array):
		
		for i, trainer in enumerate(self.trainers):
			key, key_init, key_train = jr.split(key, 3)
			new_train_state = trainer.initialize(key_init)
			if i:
				tf = self.transform_fns[i-1] if isinstance(self.transform_fns, list) else self.transform_fns 
				train_state = tf(train_state, new_train_state)
			else:
				train_state = new_train_state
			train_state = trainer.train_(train_state, key_train)

		return train_state

