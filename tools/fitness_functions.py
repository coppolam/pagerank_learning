#!/usr/bin/env python3
"""
Collection of fitness functions
@author: Mario Coppola, 2020
"""

import numpy as np
import networkx as nx
from tools import matrixOperations as matop

rangesensor = 1.8
id_column = 1

def get_positions(log):
	'''Returns the number of robots and an array of x and y positions'''
	robots = int(log[:,id_column].max())
	p = log[0:robots,(2,3)] # Positions x and y
	return robots, p


def get_distance(p_rel):
	'''Get the distance from x and y positions'''
	return np.sqrt(p_rel[:,0]**2+p_rel[:,1]**2)

def neighbors_adjacency_map(log):
	'''Returns the adjacency matrix for the swarm'''
	robots, p = get_positions(log)
	A = np.zeros([robots,robots]) # Adjacency matrix
	for r in range(0,robots):
		p_rel = p[r,:]-p
		d = (get_distance(p_rel) < rangesensor)
		A[r,:] = d
	return A

def mean_distance_to_all(log):
	'''Global fitness function to get the mean distance to all other robots in a swarm'''
	robots, p = get_positions(log)
	f = 0
	for r in range(0,robots):
		p_rel = p[r,:]-p
		d = get_distance(p_rel)
		f += d.mean()/robots
	return f

def number_of_clusters(log):
	'''Global fitness function to get number of clusters, relative to the size of the swarm'''
	A = neighbors_adjacency_map(log)
	G = nx.from_numpy_array(A)
	f = nx.components.number_connected_components(G)
	return f/robots

def largest_cluster(log):
	'''Global fitness function to get size of the largest cluster, relative to the size of the swarm'''
	A = neighbors_adjacency_map(log)
	G = nx.from_numpy_array(A)
	largest_cc = max(nx.connected_components(G), key=len)
	f = len(largest_cc)
	return f/robots

def mean_number_of_neighbors(log):
	'''Local fitness function to get the mean number of neighbors that each robot has'''
	robots, p = get_positions(log)
	f = 0
	for r in range(0,robots):
		p_rel = p[r,:]-p
		d = np.where((get_distance(p_rel) < rangesensor))[0]
		n_neighbors = d.size
		f += (n_neighbors-1) / robots
	return f
