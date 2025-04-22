from typing import Callable, Optional, Union, Literal
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn

class ResidualLinear(nn.Linear):

	#-------------------------------------------------------------------
	norm: bool
	#-------------------------------------------------------------------

	def __init__(self, in_features: Union[int, Literal["scalar"]], 
				 out_features: Union[int, Literal["scalar"]], use_bias: bool=True, 
				 norm: bool=True, *, key: jax.Array):
		super().__init__(in_features, out_features, use_bias, key=key)
		self.norm = norm

	#-------------------------------------------------------------------

	def __call__(self, x: jax.Array, h: jax.Array):
		
		y = super().__call__(x)
		h = h + y
		if self.norm:
			h = h / (jnp.linalg.norm(h)+1e-8)
		return h

	#-------------------------------------------------------------------

class ResidualMLP(nn.MLP):

	#-------------------------------------------------------------------
	norm: bool
	#-------------------------------------------------------------------

	def __init__(self, in_size: Union[int, Literal["scalar"]], 
				 out_size: Union[int, Literal["scalar"]], width_size: int, 
				 depth: int, activation: Callable=lambda x: x, 
				 final_activation: Callable=lambda x: x, use_bias: bool = True, 
				 use_final_bias: bool = True, *, key: jax.Array, norm: bool=True):
		super().__init__(in_size, out_size, width_size, depth, activation, 
						 final_activation, use_bias, use_final_bias, key=key)
		self.norm = norm

	#-------------------------------------------------------------------

	def __call__(self, x: jax.Array, h: jax.Array)->jax.Array:

		y = super().__call__(x)
		h = h + y
		if self.norm:
			h = h / (jnp.linalg.norm(h)+1e-8)
		return h

class RNN(eqx.Module):
	"""
	Simplest recurrent networks
	"""
	#-------------------------------------------------------------------
	lin: nn.Linear
	activation_fn: Callable
	#-------------------------------------------------------------------
	def __init__(self, hidden_dims: int, input_dims: int, *, key: jax.Array, 
			     activation_fn: Callable=jnn.tanh):
	
		self.lin = nn.Linear(hidden_dims+input_dims, hidden_dims, key=key)
		self.activation_fn = activation_fn

	#-------------------------------------------------------------------
	def __call__(self, x: jax.Array, h: jax.Array):
		
		return self.activation_fn(
			self.lin(
				jnp.concatenate([x,h], axis=-1)
			)
		)

class MGU(eqx.Module):
	
	"""
	Minimal Gated Unit
	https://arxiv.org/pdf/1603.09420.pdf
	"""
	#-------------------------------------------------------------------
	Wh: nn.Linear
	Wf: nn.Linear
	#-------------------------------------------------------------------

	def __init__(self, input_dims: int, hidden_dims: int, *, key: jax.Array):

		kh, kf = jr.split(key)
		self.Wh = nn.Linear(input_dims+hidden_dims, hidden_dims, key=kh)
		self.Wf = nn.Linear(input_dims+hidden_dims, hidden_dims, key=kf)

	#-------------------------------------------------------------------

	def __call__(self, x: jax.Array, h: jax.Array):
		
		xh = jnp.concatenate([x,h], axis=-1)
		f = jnn.sigmoid(self.Wf(xh))
		fhx = jnp.concatenate([f*h, x], axis=-1)
		h_ = jnn.tanh(self.Wh(fhx))
		return (1-f)*h + f*h_


class LRNN(eqx.Module):
	
	"""
	Linear Recurrent Neural Network
	"""
	#-------------------------------------------------------------------
	# Parameters:
	W_ih: nn.Linear
	W_hh: nn.Linear
	# Statics:
	
	#-------------------------------------------------------------------

	def __init__(self, in_features: int, hidden_features: int, *, key: jax.Array):
		
		kih, khh = jr.split(key)
		self.W_ih = nn.Linear(in_features, hidden_features, key=kih)
		self.W_hh = nn.Linear(hidden_features, hidden_features, key=khh)

	#-------------------------------------------------------------------

	def __call__(self, x: jax.Array, h: jax.Array):
		
		return self.W_hh(h) + self.W_ih(x)
		

class FiLM(eqx.Module):
	
	"""
	paper: https://arxiv.org/pdf/1709.07871.pdf
	"""
	#-------------------------------------------------------------------
	func: Union[nn.Linear, nn.MLP, Callable]
	#-------------------------------------------------------------------

	def __init__(self, x_dims: int, y_dims: int, *, key: jax.Array, func: Optional[Callable]=None):
		
		if func is None:
			self.func = nn.Sequential(
				[
					nn.Linear(y_dims, x_dims*2, key=key), 
					nn.Lambda(lambda x: (x[:x_dims], x[x_dims:]))
				]
			)
		else:
			self.func = func

	#-------------------------------------------------------------------

	def __call__(self, x: jax.Array, y: jax.Array):
		
		g, b = jax.vmap(self.func)(y)
		return g * x + b


if __name__ == '__main__':
	key = jr.PRNGKey(1)
	kx, ky, km = jr.split(key, 3)
	x = jr.normal(kx, (32, 10))
	y = jr.normal(ky, (32, 5))
	f = FiLM(10, 5, key=km)
	x_ = f(x, y)
	print(x_.shape)