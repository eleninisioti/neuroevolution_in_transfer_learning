import jax
import equinox as eqx
import jax.random as jr
from typing import Optional, Tuple, NamedTuple
from jaxtyping import Float, Array, PyTree, Int, ArrayLike

class Node(NamedTuple):
	#-------------------------------------------------------------------
	h: Float[Array, "N Dn"]
	m: Optional[Float[Array, "N"]]=None
	pholder: Optional[PyTree]=None
	#-------------------------------------------------------------------

class Edge(NamedTuple):
	#-------------------------------------------------------------------
	A: Float[Array, "N N ..."]
	e: Optional[Float[Array, "N N De"]]=None
	pholder: Optional[PyTree]=None
	#-------------------------------------------------------------------

class Graph(NamedTuple):
	#-------------------------------------------------------------------
	nodes: Node
	edges: Edge
	global_: Optional[PyTree]=None
	pholder: Optional[PyTree]=None
	#-------------------------------------------------------------------
	@property
	def h(self):
		return self.nodes.h
	#-------------------------------------------------------------------
	@property
	def A(self):
		return self.edges.A
	#-------------------------------------------------------------------
	@property
	def E(self):
		return self.edges.e
	#-------------------------------------------------------------------
	@property
	def e(self):
		return self.edges.e
	#-------------------------------------------------------------------
	@property
	def X(self):
		return self.nodes.h
	#-------------------------------------------------------------------
	@property
	def N(self):
		return self.nodes.h.shape[0]
	#-------------------------------------------------------------------
	def replace_at(self, where, val):
		return eqx.tree_at(where, self, val)
	#-------------------------------------------------------------------
	def replace(self, **kwargs):
		return eqx.tree_at(lambda x: [getattr(x, k) for k in kwargs.keys()],
						   self,
						   list(kwargs.values()))
	#-------------------------------------------------------------------



class GraphModule(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	#-------------------------------------------------------------------

	def __call__(self, graph: Graph, key: jax.Array)->Graph:
		
		raise NotImplementedError

	#-------------------------------------------------------------------


class IterativeGraphModule(GraphModule):

	#-------------------------------------------------------------------

	def rollout(self, graph: Graph, key: jax.Array, steps: int)->Tuple[Graph, Graph]:

		def step(carry, x):
			graph, key = carry
			key, subkey = jr.split(key)
			graph = self.__call__(graph, subkey)
			return [graph, key], graph

		[graph, _], graphs = jax.lax.scan(step, [graph, key], None, steps)
		return graph, graphs

	#-------------------------------------------------------------------

if __name__ == '__main__':
	import jax.numpy as jnp
	n = Node(h = jnp.ones((10, 2)))
	e = Edge(A=jnp.zeros((10, 10)))
	G = Graph(nodes=n, edges=e)
	G = G.replace(h=2)
	print(G)


		