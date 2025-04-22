from typing import Callable, Union, Optional
from jaxtyping import Array, Float
from src.gnn.base import Graph, Node, Edge
from src.nn.layers import RNN
import os

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


#=======================NETWORKX UTILS===========================

def to_networkx(graph: Graph)->nx.DiGraph:
	if graph.edges.A is not None:
		if graph.nodes.m is not None:
			m = graph.nodes.m.astype(bool)
			A = graph.edges.A[m][:, m].astype(int)
			return nx.from_numpy_array(A, create_using=nx.DiGraph)
		else: 
			A = graph.edges.A.astype(int)
			return nx.from_numpy_array(A, create_using=nx.DiGraph)
	else:
		raise NotImplementedError


def from_networkx(graph: Union[nx.Graph, nx.DiGraph], 
				  node_features: Optional[int]=1, 
				  edge_features: Optional[int]=1,
				  weight_field: Optional[str]="w")->Graph:
	N = len(graph.nodes)
	h = jnp.zeros((N, node_features))
	nodes = Node(h=h)
	A = nx.adjacency_matrix(graph, weight=weight_field)#type:ignore
	e = jnp.zeros((N, N, edge_features))
	edges = Edge(e=e, A=A)
	return Graph(nodes=nodes, edges=edges)

def from_graphml(file: str, node_features: int=1, edge_features: int=1):
	assert os.path.isfile(file), file
	nxG = nx.read_graphml(file)
	return from_networkx(nxG, node_features=node_features, edge_features=edge_features)


#========================VIZ UTILS=================================

def viz_graph(graph: Graph, **kwargs):
	nxg = to_networkx(graph)
	nx.draw_kamada_kawai(nxg, node_size=20, **kwargs)




