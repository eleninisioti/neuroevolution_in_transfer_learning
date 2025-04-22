from typing import Optional
from jaxtyping import Float, Array
from src.gnn.base import Graph, Node, Edge
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn



def stochastic_block_model(
	N: int,
	C: int,
	P: Float[Array, "C C"],
	key: jax.Array,
	pC: Optional[Float[Array, "C"]]=None,
	sort: bool=False)->Float[Array, "N N"]:

	kC, kA = jr.split(key)

	if pC is None:
		classes = jr.randint(kC, (N,), 0, C)
	else:
		classes = jr.choice(kC, jnp.arange(C), shape=(N,), p=pC)
	if sort:
		classes = jnp.sort(classes)
	pA = P[classes] # N x C
	pA = pA[:, classes] # N x C
	A = (jr.uniform(kA, (N, N)) < pA).astype(float)
	return A



if __name__ == '__main__':
	
	N = 1_000
	C = 3
	pii = .6
	pij = (1-pii)/(C-1)
	P = jnp.array([[pii, pij, pij],
				   [pij, pii, pij],
				   [pij, pij, pii]])
	pC = jnp.array([.2, .5, .3])
	key = jr.PRNGKey(1)	

	A = stochastic_block_model(N, C, P, key, sort=True, pC=pC)
	import matplotlib.pyplot as plt

	plt.imshow(A)
	plt.show()