from jax._src.core import Value
from base.tasks.base import BaseTask, TaskParams

from typing import Callable, Iterable, Union, Tuple
from jaxtyping import Float, PyTree
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx

Data = PyTree
Params = PyTree

stack_trees = lambda trees: jax.tree_map(lambda ts: jnp.stack(trees), trees)

class CurricTask(BaseTask):
	
	"""
	"""
	#-------------------------------------------------------------------
	tasks: Iterable[Union[BaseTask, Callable]]
	mode: str
	#-------------------------------------------------------------------

	def __init__(self, tasks: Iterable[Union[BaseTask, Callable]]):

		self.gen_switch = 350000
		self.tasks = tasks

	#-------------------------------------------------------------------

	def __call__(self, params: Params, key: jax.Array, task_params: TaskParams = None,
			current_gen: int=0) -> Tuple[Float, Data]:
		
		return self._eval_all(params, key, task_params, current_gen)


	#-------------------------------------------------------------------

	def _eval_all(self, params: Params, key: jax.Array, task_params: TaskParams = None, current_gen=0) -> Tuple[Float, Data]:
		"""evaluate params on all tasks """
		datas = []
		fit_sum = 0.
		key, subkey = jr.split(key)

		fit, info, policy_states, task_params = jax.lax.cond(current_gen <self.gen_switch, lambda x: self.tasks[0](x, subkey, task_params),lambda x: self.tasks[1](x, subkey, task_params), params)

		#fit, data = task(params, subkey, task_params)

		#datas.append(data)
		return fit, info, policy_states, task_params