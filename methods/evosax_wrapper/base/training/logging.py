import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from typing import Tuple, TypeAlias, Callable, Optional
from jax.experimental import io_callback
import equinox as eqx
import os
import pickle
TrainState: TypeAlias = PyTree[...]
Data: TypeAlias = PyTree[...]

class Logger:

	#-------------------------------------------------------------------

	def __init__(
		self, 
		wandb_log: bool,
		metrics_fn: Callable[[TrainState, Data], Tuple[Data, Data, int]],
		ckpt_dir: Optional[str]=None,
		aim_freq: int=10,
		ckpt_freq: int=100,
		dev_steps: int=0,
		verbose: bool=False):



		self.wandb_log = wandb_log
		self.metrics_fn = metrics_fn
		self.ckpt_dir = ckpt_dir
		self.ckpt_freq = ckpt_freq
		self.aim_freq = aim_freq
		self.epoch = [0]
		self.verbose = verbose
		self.dev_steps = dev_steps

	#-------------------------------------------------------------------




	def log(self, state: TrainState, data: Data, task_params: jnp.array):



		#self.save_genes(genes, fitnesses, jnp.array(epoch))

		#self.save_best_model(ckpt_data, jnp.array(epoch))

		for dev_step in range(self.dev_steps+2):
			current_dev =  jax.tree_map(lambda x: x[data["best_indiv"], 0, dev_step,...], data["interm_policies"])
			self.save_chkpt(current_dev, state.gen_counter, jnp.array(dev_step))

		weights = current_dev.weights

		# number of edges
		non_zero_count = jnp.count_nonzero(weights)

		# Identify non-zero rows
		non_zero_rows = jnp.any(weights != 0, axis=1)

		# Identify non-zero columns
		non_zero_columns = jnp.any(weights != 0, axis=0)

		# Count unique indexes for which row or column is non-zero
		unique_indexes_count = jnp.count_nonzero(non_zero_rows) + jnp.count_nonzero(non_zero_columns)

		num_edges= non_zero_count
		num_nodes= unique_indexes_count
		jax.lax.cond(state.gen_counter%self.aim_freq==0, lambda data: self.metrics_fn(state, data, task_params, num_nodes, num_edges), lambda data: None, data)





	def save_genes(self, data: dict, fitnesses: list, epoch: int):

		def save(d):
			data, fitnesses = d
			assert self.ckpt_dir is not None

			save_dir = self.ckpt_dir + "/evo_anal"
			if not os.path.exists(save_dir):
				os.makedirs(save_dir, exist_ok=True)
			with open(save_dir + "/genes.pkl", "wb") as f:
				pickle.dump((data, fitnesses), f)

			return d

		def tap_save(data, fitnesses):
			io_callback(lambda d, *_: save(d), (data, fitnesses), (data, fitnesses))
			return None

		if self.ckpt_dir is not None:
			jax.lax.cond(
				(jnp.mod(epoch, self.ckpt_freq)) == 0,
				lambda data: tap_save(data, fitnesses),
				lambda data: None,
				data
			)


	def save_evo_anal(self, phylo_depth: dict, epoch: int):

		def save(d):
			data, epoch = d
			assert self.ckpt_dir is not None

			save_dir = self.ckpt_dir + "/evo_anal"
			if not os.path.exists(save_dir):
				os.makedirs(save_dir, exist_ok=True)
			with open(save_dir + "/phylo_depth_" + str(epoch) + ".pkl", "wb") as f:
				pickle.dump(data, f)
			return d

		def tap_save(data, epoch):
			io_callback(lambda d, *_: save(d), (data, epoch), (data, epoch))
			return None

		if self.ckpt_dir is not None:
			jax.lax.cond(
				(jnp.mod(epoch, self.ckpt_freq)) == 0,
				lambda data: tap_save(data, epoch),
				lambda data: None,
				phylo_depth
			)


	def save_best_model(self, data: dict, epoch: int):

		def save(d):
			data, epoch= d
			assert self.ckpt_dir is not None
			save_dir= self.ckpt_dir + "/best_model"
			if not os.path.exists(save_dir):
				os.makedirs(save_dir, exist_ok=True)

			file = save_dir + "/ckpt_" + str(epoch) + ".eqx"
			if self.verbose:
				print("saving data at: ", file)
			eqx.tree_serialise_leaves(file, data)

			save_dir = self.ckpt_dir + "/evo_anal"
			if not os.path.exists(save_dir):
				os.makedirs(save_dir, exist_ok=True)

			return d


		def tap_save(data, epoch):
			io_callback(lambda d, *_: save(d), (data, epoch), (data, epoch))
			return None

		if self.ckpt_dir is not None:
			jax.lax.cond(
				(jnp.mod(epoch, self.ckpt_freq)) == 0,
				lambda data: tap_save(data, epoch),
				lambda data: None,
				data
			)

	#-------------------------------------------------------------------

	def _log(self, data: dict):

		def call_wandb(data):
			wandb.log(data)
			return data

		io_callback(
			lambda d, *_: call_wandb(d), data, data
		)

	#-------------------------------------------------------------------


	def save_task_checkpoint(self, data: dict, task: int, dev_step: int):

		def save(d):
			data, epoch, dev_step = d
			assert self.ckpt_dir is not None
			save_dir= self.ckpt_dir + "/task_checkpoints/task_" + str(task)
			if not os.path.exists(save_dir):
				os.makedirs(save_dir, exist_ok=True)

			file = save_dir + "/dev_" + str(dev_step) + ".pkl"

			if self.verbose:
				print("saving data at: ", file)
			with open(file, "wb") as f:
				pickle.dump( data,f)
			return d
			#eqx.tree_serialise_leaves(file, data)

		def tap_save(data, epoch, dev_step):

			io_callback(lambda d, *_: save(d), (data, epoch, dev_step), (data, epoch, dev_step))
			return None

		if self.ckpt_dir is not None:
			jax.lax.cond(
				(jnp.mod(epoch, self.ckpt_freq))==0,
				lambda data : tap_save(data, epoch, dev_step),
				lambda data : None,
				data
			)


	def save_chkpt(self, data: dict, epoch: int, dev_step: int):

		def save(d):
			data, epoch, dev_step = d
			assert self.ckpt_dir is not None
			save_dir= self.ckpt_dir + "/all_info/gen_" + str(epoch)
			if not os.path.exists(save_dir):
				os.makedirs(save_dir, exist_ok=True)

			file = save_dir + "/dev_" + str(dev_step) + ".pkl"

			if self.verbose:
				print("saving data at: ", file)
			with open(file, "wb") as f:
				pickle.dump( data,f)
			return d
			#eqx.tree_serialise_leaves(file, data)

		def tap_save(data, epoch, dev_step):

			io_callback(lambda d, *_: save(d), (data, epoch, dev_step), (data, epoch, dev_step))
			return None

		if self.ckpt_dir is not None:
			jax.lax.cond(
				(jnp.mod(epoch, self.ckpt_freq))==0,
				lambda data : tap_save(data, epoch, dev_step),
				lambda data : None,
				data
			)

	#-------------------------------------------------------------------

	def wandb_init(self, project: str, config: dict, **kwargs):
		if self.wandb_log:
			wandb.init(project=project, config=config, **kwargs)

	#-------------------------------------------------------------------

	def wandb_finish(self, *args, **kwargs):
		if self.wandb_log:
			wandb.finish(*args, **kwargs)
