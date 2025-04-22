from functools import partial
from src.gnn.base import GraphModule, Graph, Node, Edge
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
from typing import Callable, Optional, TypeAlias
from jaxtyping import Float, Array, Int

AdjacencyMatrix: TypeAlias = Float[Array, "N N"] | Int[Array, "N N"]

def in_degrees(A: AdjacencyMatrix)->Float[Array, "N"]:
	return A.sum(0)

def out_degrees(A: AdjacencyMatrix)->Float[Array, "N"]:
	return A.sum(1)

def degrees(A: AdjacencyMatrix)-> Float[Array, "N"]:
	return out_degrees(A) + in_degrees(A)

def avg_in_neighbor_in_degree(A: AdjacencyMatrix)-> Float[Array, "N"]:
	d = in_degrees(A)
	return jnp.dot(A.T, d)

def avg_in_neighbor_out_degree(A: AdjacencyMatrix)-> Float[Array, "N"]:
	d = out_degrees(A)
	return jnp.dot(A.T, d)

def avg_out_neighbor_out_degree(A: AdjacencyMatrix)-> Float[Array, "N"]:
	d = out_degrees(A)
	return jnp.dot(A, d)

def avg_out_neighbor_in_degree(A: AdjacencyMatrix)-> Float[Array, "N"]:
	d = in_degrees(A)
	return jnp.dot(A, d)

def adjacency_powers(A: AdjacencyMatrix, k: int)->Float[Array, "P N N"]:
	Af, As = jax.lax.scan(
		lambda Ai, i: (jnp.matmul(Ai, A),)*2, #type:ignore
		A, jnp.arange(k) 
	)
	return jnp.concatenate([As, Af[None]], axis=0)

def count_k_cycles(A: jax.Array, k: Int):
	# Compute the kth power of the adjacency matrix
	A = jax.lax.fori_loop(0, k-1, lambda i, Ai: jnp.matmul(Ai, A), A)
	# Count the number of k-cycles for each node
	cycle_counts = jnp.diag(A)
	return cycle_counts

def count_total_k_cycles(A, k):
	# Compute the kth power of the adjacency matrix
	A = jax.lax.fori_loop(0, k-1, lambda i, Ai: jnp.matmul(Ai, A), A)

	# The number of k-cycles in the whole graph is half the trace of Ak
	total_cycles = jnp.trace(A) // 2  # // 2 to avoid double counting
	return total_cycles

def weighted_communicability(A: AdjacencyMatrix, W: AdjacencyMatrix):
	"""
	Args: 
	A (array): Adjacency matrix
	W (array): Weight matrix
	"""
	D = jnp.sqrt(jnp.diag(degrees(A)))
	C = jnp.exp(D @ W @ D)
	return C

def communicability(A: jax.Array):
	D = jnp.sqrt(jnp.diag(degrees(A)))
	C = jnp.exp(D @ A @ D)
	return C

def controllability_gramian(A: jax.Array, B: jax.Array, T: int):
	"""Returns the controllability gramian of graph up to T timesteps"""
	def _t_term(At, B):
		return At @ B @ B.T @ At.T
	_, A_ts = jax.lax.scan(lambda a, _: (a @ A, a), A, None, T)
	return jnp.sum(jax.vmap(_t_term, in_axes=(0,None))(A_ts, B), axis=0)


	#-------------------------------------------------------------------


if __name__ == '__main__':
	k = jr.PRNGKey(1)
	#A = jr.randint(k, (10, 10), 0, 2).astype(float)
	A = jnp.zeros((10,10)).at[:, :].set(1.)
	B = jnn.one_hot(jnp.arange(10), 10, axis=-1)[...,None]
	ctrb = partial(controllability_gramian, T=10)
	Ws = jax.vmap(ctrb, in_axes=(None,0))(A, B)
	# print(W.shape)
	print([float(jnp.trace(W)) for W in Ws])
	# print(jnp.min(jnp.linalg.eigvals(W)))
