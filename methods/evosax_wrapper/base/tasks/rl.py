from ast import Call
from typing import Callable, NamedTuple, Optional, Tuple, TypeAlias, Union
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from stepping_gates import envs as gate_envs
#from simple import envs as simple_envs
from brax import envs as brax_envs
from ecorobot import envs as ecorobot_envs
from brax.envs import Env
#from brax import envs as brax_envs
from jaxtyping import Float, PyTree

Params: TypeAlias = PyTree
TaskParams: TypeAlias = PyTree
EnvState: TypeAlias = PyTree
Action: TypeAlias = jax.Array
PolicyState: TypeAlias = PyTree
BraxEnv: TypeAlias = Env

class State(NamedTuple):
	env_state: EnvState
	policy_state: PolicyState
	#action: jnp.array


#=======================================================================
#=======================================================================
#=======================================================================
class EcorobotTask(eqx.Module):
	"""
	"""
	#-------------------------------------------------------------------
	env: BraxEnv
	statics: PyTree[...]
	max_steps: int	
	num_tasks: int
	current_task: int
	reward_for_solved: float
	data_fn: Callable[[PyTree], dict]
 
	#-------------------------------------------------------------------
	def __init__(
		self, 
		statics: PyTree[...],
		env: Union[str, BraxEnv],
		max_steps: int,
		backend: str="mjx",
		data_fn: Callable=lambda x: x, 
		env_kwargs: dict={}):

		if isinstance(env, str):
			self.env = ecorobot_envs.get_environment(env_name=env, backend=backend, **env_kwargs)
		else:
			self.env = env

		self.statics = statics
		self.max_steps = max_steps
		self.data_fn = data_fn
		self.num_tasks = 1
		self.reward_for_solved = 5000
		self.current_task = 0


	def __call__(
		self, 
		params: Params, 
		key: jax.Array, 
		task_params: Optional[TaskParams]=None,
			current_gen: int=0)->Tuple[Float, PyTree]:

		_, _, data, policy_states= self.rollout(params, key)
		return jnp.sum(data["reward"]), data, policy_states, 0.0

 
	def initialize(self, key: jax.Array, target_function=None) -> EnvState:

		return self.env.reset(key)

	def rollout(
		self, 
		params: Params, 
		key: jax.Array, 
		task_params: Optional[TaskParams]=None)->Tuple[State, State, dict]:

		init_env_key, init_policy_key, rollout_key = jr.split(key, 3)
		policy = eqx.combine(params, self.statics)

		policy_state, policy_states = policy.initialize(init_policy_key)
		env_state = self.initialize(init_env_key)
		init_state = State(env_state=env_state, policy_state=policy_state)

		obs_size = self.env.observation_size
		action_size = self.env.action_size

		def env_step(carry, x):
			state, key = carry
			key, _key = jr.split(key)
			action, policy_state = policy(state.env_state.obs, state.policy_state, _key,obs_size=obs_size,action_size=action_size)
			env_state = self.env.step(state.env_state, action)
			new_state = State(env_state=env_state, policy_state=policy_state)
			
			return [new_state, key], (state, action)

		[state, _], (states, actions) = jax.lax.scan(env_step, [init_state, rollout_key], None, self.max_steps)	
		data = {"policy_states": states.policy_state, "obs": states.env_state.obs}
		data = self.data_fn(data)
		data["reward"] = states.env_state.reward
		data["actions"]  = actions
		return state, states, data, policy_states

class BraxTask(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	env: BraxEnv
	statics: PyTree[...]
	max_steps: int

	num_tasks: int
	current_task: int
	reward_for_solved: float

	data_fn: Callable[[PyTree], dict]
	#-------------------------------------------------------------------

	def __init__(
		self, 
		statics: PyTree[...],
		env: Union[str, BraxEnv],
		max_steps: int,
		backend: str="positional",
		data_fn: Callable=lambda x: x, 
		env_kwargs: dict={}):
		
		if isinstance(env, str):
			self.env = brax_envs.get_environment(env, backend=backend, **env_kwargs)
		else:
			self.env = env

		self.statics = statics
		self.max_steps = max_steps
		self.data_fn = data_fn
		self.num_tasks = 1
		#self.num_tasks = self.env.num_tasks
		self.reward_for_solved = 5000
		self.current_task = 0

	#-------------------------------------------------------------------

	def __call__(
		self, 
		params: Params, 
		key: jax.Array, 
		task_params: Optional[TaskParams]=None,
			current_gen: int=0)->Tuple[Float, PyTree]:

		_, _, data, policy_states= self.rollout(params, key)
		return jnp.sum(data["reward"]), data, policy_states, 0.0

	#-------------------------------------------------------------------

	def rollout(
		self, 
		params: Params, 
		key: jax.Array, 
		task_params: Optional[TaskParams]=None)->Tuple[State, State, dict]:
		
		init_env_key, init_policy_key, rollout_key = jr.split(key, 3)
		policy = eqx.combine(params, self.statics)
		
		policy_state, policy_states = policy.initialize(init_policy_key)
		env_state = self.initialize(init_env_key)
		init_state = State(env_state=env_state, policy_state=policy_state)
		obs_size = self.env.observation_size
		action_size = self.env.action_size
		def env_step(carry, x):
			state, key = carry
			key, _key = jr.split(key)
			action, policy_state = policy(state.env_state.obs, state.policy_state, _key,obs_size=obs_size,action_size=action_size)
			env_state = self.env.step(state.env_state, action)
			new_state = State(env_state=env_state, policy_state=policy_state)
			
			return [new_state, key], (state, action)

		[state, _], (states, actions) = jax.lax.scan(env_step, [init_state, rollout_key], None, self.max_steps)
		data = {"policy_states": states.policy_state, "obs": states.env_state.obs}
		data = self.data_fn(data)
		first_done = jnp.argmax(states.env_state.done)
		indexes = jnp.arange(states.env_state.reward.shape[0])
		data["reward"] = jnp.where(indexes > first_done, 0, states.env_state.reward)
		data["actions"]  = actions
		data["reward"] = states.env_state.reward
		#data["reward"] = states.env_state.reward*(1-states.env_state.done) # do not take into account rewards from steps where the episode is done
		#data["actions"] = actions
		return state, states, data, policy_states

	#-------------------------------------------------------------------

	def step(self, *args, **kwargs):
		return self.env.step(*args, **kwargs)

	def reset(self, *args, **kwargs):
		return self.env.reset(*args, **kwargs)

	#-------------------------------------------------------------------

	def initialize(self, key:jax.Array)->EnvState:
		
		return self.env.reset(key)




class GatesTask(eqx.Module):
	"""
	"""
	# -------------------------------------------------------------------
	env: BraxEnv
	statics: PyTree[...]
	max_steps: int
	num_tasks: int
	current_task: int
	data_fn: Callable[[PyTree], dict]

	# -------------------------------------------------------------------

	def __init__(
			self,
			statics: PyTree[...],
			env: Union[str, BraxEnv],
			max_steps: int,
			data_fn: Callable = lambda x: x,
			env_kwargs: dict = {}):

		if isinstance(env, str):
			self.env = gate_envs.get_environment(env_name=env,**env_kwargs)
		else:
			self.env = env
		self.num_tasks = self.env.num_tasks
		self.statics = statics
		self.max_steps = max_steps
		self.data_fn = data_fn
		self.current_task = self.env.current_task

	# -------------------------------------------------------------------

	def __call__(
			self,
			params: Params,
			key: jax.Array,
			task_params: Optional[TaskParams] = None,
	current_gen: int=0) -> Tuple[Float, PyTree]:

		state, _, data, policy_states, actions = self.rollout(params, key, task_params, current_gen)
		task_params = state.env_state.info["current_task"]
		return jnp.sum(data["reward"]), data, policy_states, task_params

	# -------------------------------------------------------------------

	def get_obs_size(self, task):
		return self.env.get_obs_size(task)

	def show_rollout(self, states,save_dir, filename):
		self.env.show_rollout(states,save_dir, filename)

	def get_action_size(self, task):
		return self.env.get_action_size(task)
	def rollout(
			self,
			params: Params,
			key: jax.Array,
			task_params: Optional[TaskParams] = None,
	current_gen: int=0) -> Tuple[State, State, dict]:

		init_env_key, init_policy_key, rollout_key = jr.split(key, 3)
		policy = eqx.combine(params, self.statics)

		policy_state, policy_states = policy.initialize(init_policy_key,current_gen=current_gen)
		env_state = self.initialize(init_env_key, task_params)
		init_state = State(env_state=env_state, policy_state=policy_state)


		obs_size = self.env.get_obs_size(task_params)
		action_size = self.env.get_action_size(task_params)

		def env_step(carry, x):
			state, key = carry
			key, _key = jr.split(key)
			action, policy_state = policy(state.env_state.obs, state.policy_state, _key, obs_size=obs_size,action_size=action_size)
			env_state = self.env.step(state.env_state, action)
			new_state = State(env_state=env_state, policy_state=policy_state)

			return [new_state, key], (new_state, action)

		
		"""
        states = []
		state = init_state
		for step in range(self.max_steps):
			key, _key = jr.split(key)
			action, policy_state = policy(state.env_state.obs, state.policy_state, _key)
			env_state = self.env.step(state.env_state, action)
			state = State(env_state=env_state, policy_state=policy_state)
			states.append(state)
		"""
		





		[state, _], (states, actions) = jax.lax.scan(env_step, [init_state, rollout_key], None, self.max_steps)
		data = {"policy_states": states.policy_state, "obs": states.env_state.obs}
		import numpy as onp
		data = self.data_fn(data)
		#reward = jnp.where(states.env_state.done, 0, states.env_state.reward)
		#reward = jnp.where(states.env_state.done, 0.0, 0.0)
		#first_done = jnp.argmax(states.env_state.done)
		#indexes = jnp.arange(states.env_state.reward.shape[0])
		data["reward"] = states.env_state.reward
		first_done = jnp.argmax(states.env_state.done)
		indexes = jnp.arange(states.env_state.reward.shape[0])
		data["reward"] = states.env_state.reward
		data["reward"] = jnp.where(indexes > first_done, 0, states.env_state.reward)
		def concat_fn(x, y):
			return jnp.concatenate([x, y], axis=0)

		def expand_dims_fn(x):
			return jnp.expand_dims(x, axis=0)

		#policy_states = jax.tree_map(concat_fn,  policy_states,jax.tree_map(expand_dims_fn, policy_state) )

		return state, states, data, policy_states, actions


	def render(self, states, actions):
		trajectory = ""
		for idx in range(len(states)):
			trajectory += "Step:" + str(idx) + self.env.render(states[idx].env_state, actions[idx]) + "/n"
		return trajectory


	# -------------------------------------------------------------------

	def step(self,state, action):
		return self.env.step(state, action)

	def reset(self, key, task):
		return self.env.reset(key, task)

	# -------------------------------------------------------------------

	def initialize(self, key: jax.Array, target_function) -> EnvState:

		return self.env.reset(key, target_function)



#-------------------------------------------------------------------


#=======================================================================
#=======================================================================
#=======================================================================



#=======================================================================
#=======================================================================
#=======================================================================

class RandomDiscretePolicy(eqx.Module):
	"""
	"""
	#-------------------------------------------------------------------
	n_actions: int
	#-------------------------------------------------------------------
	def __init__(self, n_actions: int):
		self.n_actions = n_actions
	#-------------------------------------------------------------------
	def __call__(self, env_state: EnvState, policy_state: PolicyState, key: jax.Array):
		return jr.randint(key, (), 0, self.n_actions), None
	#-------------------------------------------------------------------
	def initialize(self, *args, **kwargs):
		return None

class RandomContinuousPolicy(eqx.Module):
	"""
	"""
	#-------------------------------------------------------------------
	action_dims: int
	#-------------------------------------------------------------------
	def __init__(self, action_dims: int):
		self.action_dims = action_dims
	#-------------------------------------------------------------------
	def __call__(self, env_state: EnvState, policy_state: PolicyState, key: jax.Array):
		return jr.normal(key, (self.action_dims,)), None
	#-------------------------------------------------------------------
	def initialize(self, *args, **kwargs):
		return None

class StatefulPolicyWrapper(eqx.Module):
	"""
	Wrapper adding a policy state to the signature call of a stateless policy
	"""
	#-------------------------------------------------------------------
	policy: Union[PyTree[...], Callable[[EnvState, jax.Array], Action]]
	#-------------------------------------------------------------------
	def __init__(self, policy: Union[PyTree[...], Callable[[EnvState, jax.Array], Action]]):
		self.policy = policy
	#-------------------------------------------------------------------
	def __call__(self, env_state, policy_state, key):
		action = self.policy(env_state, key)
		return action, None
	#-------------------------------------------------------------------
	def initialize(self, *args, **kwargs):
		return None


ENV_SPACES = {
	"CartPole-v1": (4, 2, "discrete"),
	"halfcheetah": (17, 6, "continuous"),
	"ant": (27, 8, "continuous"),
	"walker2d": (17, 6, "continuous"),
	"inverted_pendulum": (4, 1, "continuous"),
	'inverted_double_pendulum': (8, 1, "continuous"),
	"hopper": (11, 3, "continuous")
}
		

if __name__ == '__main__':
	policy = RandomDiscretePolicy(3)
	params, statics = eqx.partition(policy, eqx.is_array)
	task = GymnaxTask(statics, "CartPole-v1")
	rews, data = task(params, jr.PRNGKey(10))
	print(rews)




