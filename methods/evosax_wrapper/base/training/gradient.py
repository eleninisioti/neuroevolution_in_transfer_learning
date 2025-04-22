from src.training.base import BaseTrainer
from src.training.logging import Logger

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import optax
from typing import Callable, Iterable, Optional, Union, Tuple, NamedTuple
from jaxtyping import Float, PyTree

Params = PyTree[...]
Data = PyTree[...]

class TrainState(NamedTuple):
	params: Params
	opt_state: optax.OptState
	epoch: int
	best_loss: Float
	best_params: Float

class OptaxTrainer(BaseTrainer):
	
	"""
	"""
	#-------------------------------------------------------------------
	optimizer: optax.GradientTransformation
	loss_fn: Callable[[Params, Optional[jr.PRNGKeyArray]], Tuple[Float, Data]]
	initializer: Callable[[jr.PRNGKeyArray], Params]
	#-------------------------------------------------------------------

	def __init__(
		self, 
		epochs: int,
		optimizer: Union[optax.GradientTransformation, str],
		initializer: Callable[[jr.PRNGKeyArray], Params],
		loss_fn: Callable[[Params, jr.PRNGKeyArray], Float], 
		learning_rate: Optional[float]=0.01,
		opt_kws: Optional[dict]={},
		logger: Optional[Logger]=None,
	    progress_bar: Optional[bool]=False):

		super().__init__(epochs, logger=logger, progress_bar=progress_bar)
		
		if isinstance(optimizer, str):
			OPT = getattr(optax, optimizer)
			self.optimizer = OPT(learning_rate=learning_rate, **opt_kws)
		else:
			self.optimizer = optimizer

		self.loss_fn = loss_fn
		self.initializer = initializer

	#-------------------------------------------------------------------

	def train_params(self, params: Params, key: jr.PRNGKeyArray)->Tuple[TrainState, Data]:

		state = TrainState(params=params, opt_state=self.optimizer.init(params), epoch=0,
						   best_params=params, best_loss=jnp.inf)
		return self.train(state, key) # type: ignore

	#-------------------------------------------------------------------

	def train_step(self, state: TrainState, key: jr.PRNGKeyArray, task_params: Optional[PyTree]=None) -> Tuple[TrainState, Data]:
		
		[loss, eval_data], grads = jax.value_and_grad(self.loss_fn, has_aux=True)(state.params, key)
		updates, opt_state = self.optimizer.update(grads, state.opt_state, params)
		params = optax.apply_updates(state.params, updates)
		is_best = loss<state.best_loss
		bl = jnp.where(is_best, loss, state.best_loss)
		bp = jnp.where(is_best, state.params, state.best_params)
		return (TrainState(params=params, opt_state=opt_state, epoch=state.epoch+1, best_params=bp, best_loss=bl), 
				{"loss": loss, "eval_data": eval_data})

	#-------------------------------------------------------------------

	def initialize(self, key: jr.PRNGKeyArray) -> TrainState:
		
		init_params = self.initializer(key)
		opt_state = self.optimizer.init(init_params)
		return TrainState(params=init_params, opt_state=opt_state, epoch=0, 
						  best_params=init_params, best_loss=jnp.inf)

class DataBasedOptaxTrainer(OptaxTrainer):
	#-------------------------------------------------------------------
	data_loader: Iterable
	loss_fn: Callable[[Params, Data, jax.Array], Float] #type:ignore
	#-------------------------------------------------------------------
	def __init__(self, 
				 epochs: int, 
				 data_loader: Iterable,
				 optimizer: Union[optax.GradientTransformation, str], 
				 initializer: Callable[[jr.PRNGKeyArray], Params], 
				 loss_fn: Callable[[Params, jr.PRNGKeyArray], Float], 
				 learning_rate: Optional[float] = 0.01, 
				 opt_kws: Optional[dict] = {}, 
				 logger: Optional[Logger] = None, 
				 progress_bar: Optional[bool] = False):

		super().__init__(epochs, optimizer, initializer, loss_fn, learning_rate, opt_kws, logger, progress_bar)
		self.data_loader = data_loader
	#-------------------------------------------------------------------
	def train_step(self, state: TrainState, key: jr.PRNGKeyArray, task_params: Optional[PyTree]=None) -> Tuple[TrainState, Data]:
		params = state.params
		bl = state.best_loss
		bp = state.best_params
		opt_state = state.opt_state
		total_loss = jnp.zeros(())
		for data in self.data_loader:
			[loss, _], grads = jax.value_and_grad(self.loss_fn, has_aux=True)(params, data, key)
			updates, opt_state = self.optimizer.update(grads, opt_state, params)
			is_best = loss<bl
			bl = jax.lax.cond(is_best,
							  lambda *_: loss,
							  lambda *_: bl, 0)
			bp = jax.lax.cond(is_best,
							  lambda *_: params,
							  lambda *_: bp, 0)
			params = optax.apply_updates(params, updates)
			total_loss = total_loss + loss

		return (TrainState(params=params, opt_state=opt_state, epoch=state.epoch+1, best_params=bp, best_loss=bl), 
				{"loss": total_loss/len(self.data_loader)}) #type:ignore


def default_metrics_fn(train_state, data):
	return {"loss": data["loss"]}, train_state.best_params, train_state.epoch  