#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:48:31 2019

@author: mario
"""

import networkx as nx

def draw_graph(G):
    nx.draw_networkx(G,nx.planar_layout(G))
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,nx.planar_layout(G),edge_labels=labels)
    
def make_graph(s,t,w=None):
    if w is None:
        edge_list = list(zip(s,t))
    else:
        edge_list = list(zip(s,t,w))
        
    G = nx.DiGraph()
    G.add_weighted_edges_from(edge_list)
    
    return G

