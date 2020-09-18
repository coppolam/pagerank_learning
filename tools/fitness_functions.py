#!/usr/bin/env python3
"""
Collection of fitness functions

This is useful to recreate the fitness of a run 
without re-running the simulation.

This also allows us to test how the same exact run would have
been with a different fitness function

It only pertains to tasks without environment dependent fitness.
In this case A, C1, and C2.

@author: Mario Coppola, 2020
"""

import numpy as np
import networkx as nx
from . import matrixOperations as matop

# This is how far the robots can sense in the simulation
rangesensor = 1.8

# This is the column in the log file where the ID is stored
id_column = 1

# These are the columns where the x and y positions are store
x_column, y_column = 2, 3

def get_positions(log):
	'''Returns the number of robots and an array of x and y positions'''

	# Get the number of robots (max ID in the log)
	robots = int(log[:,id_column].max())

	# Positions x and y from the log at columns 2 and 3
	p = log[0:robots,(x_column,y_column)]

	return robots, p

def get_distance(p_rel):
	'''Get the distance from a nested list of x and y positions'''
	return np.sqrt(p_rel[:,0]**2+p_rel[:,1]**2)

def neighbors_adjacency_map(log):
	'''Returns the adjacency matrix for the swarm'''
	
	# Get the number of robots and their positions
	robots, p = get_positions(log)

	# Make an adjacency matrix of their connectivity
	## Initialize
	A = np.zeros([robots,robots])
	## Fill it in
	for r in range(0,robots):
		p_rel = p[r,:]-p
		A[r,:] = (get_distance(p_rel) < rangesensor)
	
	return A, robots

def mean_distance_to_all(log):
	'''
	Global fitness function to get the mean distance 
	to all other robots in a swarm
	'''
	
	# Get the number of robots and their positions
	robots, p = get_positions(log)

	# Return a fitness = mean distance between all robots
	## Initialize the fitness f
	f = 0
	## Add to it
	for r in range(0,robots):
		d = get_distance(p[r,:]-p)
		f += d.mean()/robots
	
	return f

def number_of_clusters(log):
	'''
	Global fitness function to get number of clusters, 
	relative to the size of the swarm
	'''
	# Get an adjacency map of all robots
	A, robots = neighbors_adjacency_map(log)

	# Set up a graph G showing their connections with networkx
	G = nx.from_numpy_array(A)

	# Use networkx to get the number of connected components
	f = nx.components.number_connected_components(G)

	return 1/(f/robots)

def largest_cluster(log):
	'''
	Global fitness function to get size of the largest cluster, 
	relative to the size of the swarm
	'''
	
	# Get an adjacency map of all robots
	A, robots = neighbors_adjacency_map(log)
	
	# Set up a graph G showing their connections with networkx
	G = nx.from_numpy_array(A)
	
	# Use networkx to get the largest clusters in G
	largest_cc = max(nx.connected_components(G), key=len)

	# Get fitness = size of the largest cluster
	f = len(largest_cc)

	return f/robots

def mean_number_of_neighbors(log):
	'''
	Local fitness function to get the mean number
	of neighbors that each robot has
	'''
	# Get positions of robots
	robots, p = get_positions(log)

	# Get fitness
	## Initialize fitness
	f = 0
	
	## Add to it
	for r in range(0,robots):
    	### Distance to robots
		d = np.where((get_distance(p[r,:]-p) < rangesensor))[0]
		
		### Size of neighborhood (exclude current robot, hence the -1)
		n_neighbors = d.size - 1
		
		### Add to fitness
		f += (n_neighbors) / robots
	
	return f