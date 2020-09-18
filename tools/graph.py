#!/usr/bin/env python3
"""
Collection of graph tools based on the networkx package
to simplify certain operations

@author: Mario Coppola, 2020
"""
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import to_agraph
import networkx as nx

def print_graph(G,name="graph.eps"):
    '''Draws a pretty representation of a networkx graph'''
    A = nx.drawing.nx_agraph.to_agraph(G)
    A.layout("circo")
    A.draw(name)
    
def make_digraph(s,t,w=None):
    '''
    Makes a digraph with potential weights w
    It takes in two required arguments:
    s: A list indicating start nodes of all edges
    t: A list indicating end nodes of all edges
    '''
    # Set up a NetworkX graph
    G = nx.OrderedMultiDiGraph()

    # If weights are not specified, make the graph without weights
    if w is None:
        edge_list = list(zip(s,t))
        G.add_edges_from(edge_list)
    # Else make weighted edges
    else:
        edge_list = list(zip(s,t,w))
        G.add_weighted_edges_from(edge_list)
    return G