#!/usr/bin/env python3
"""
Collection of graph tools based on the networkx package to simplify certain operations
@author: Mario Coppola, 2020
"""
from networkx.drawing.nx_agraph import to_agraph
import networkx as nx

# Draw graph together with edge labels, takes a networkx graph as input
def print_graph(G,name):
    A = to_agraph(G)
    A.layout('dot')
    A.draw(name)  # Output a figure of the graph

# Make a digraph together with edge weights, if given
def make_digraph(s,t,w=None):
    G = nx.OrderedMultiDiGraph()
    if w is None:
        edge_list = list(zip(s,t))
        G.add_edges_from(edge_list)
    else:
        edge_list = list(zip(s,t,w))
        G.add_weighted_edges_from(edge_list)
    return G
