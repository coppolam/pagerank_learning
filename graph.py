#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:48:31 2019

Collection of graph tools based on the networkx package to simplify certain operations

@author: mario
"""

import networkx as nx
import numpy

# Draw graph together with edge labels, takes a networkx graph as input
def draw_graph(G):
    nx.draw_networkx(G,nx.planar_layout(G))
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,nx.planar_layout(G),edge_labels=labels)

# Make a digraph together with edge weights, if given
def make_digraph(s,t,w=None):
    if w is None:
        edge_list = list(zip(s,t))
    else:
        edge_list = list(zip(s,t,w))
        
    G = nx.MultiDiGraph()
    G.add_weighted_edges_from(edge_list)
    return G

# Evaluate PageRank from G matrix
def pagerank(G, tol=1e-6):
    # Iterative procedure
    n = G.shape[0] # Size of G
    pr = 1 / n * numpy.ones((1, n)) # Initialize Pagerank vector
    residual = 1 # Residual (initialize high, doesn't really matter)
    while residual >= tol:
        pr_previous = pr
        pr = numpy.matmul(pr,G) # Pagerank formula
        residual = numpy.linalg.norm(numpy.subtract(pr,pr_previous))
    return numpy.squeeze(numpy.asarray(pr))


