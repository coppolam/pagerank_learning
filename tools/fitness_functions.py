import numpy as np
import networkx as nx
from tools import matrixOperations as matop

def mean_distance_to_rest(log):
	id_column = 1
	robots = int(log[:,id_column].max())
	p = log[0:robots,(2,3)] # Positions
	f = 0
	for r in range(0,robots):
		p_rel = p[r,:]-p
		d = np.sqrt(p_rel[:,0]**2+p_rel[:,1]**2)
		f += d.mean()/robots
	return f

def number_of_clusters(log):
	id_column = 1
	robots = int(log[:,id_column].max())
	p = log[0:robots,(2,3)] # Positions
	rangesensor = 1.8
	A = np.zeros([robots,robots]) # Adjacency matrix
	for r in range(0,robots):
		p_rel = p[r,:]-p
		d = (np.sqrt(p_rel[:,0]**2+p_rel[:,1]**2) < rangesensor)
		A[r,:] = d
	G = nx.from_numpy_array(A)
	f = nx.components.number_connected_components(G)
	return f/robots

def largest_cluster(log):
	id_column = 1
	robots = int(log[:,id_column].max())
	p = log[0:robots,(2,3)] # Positions
	rangesensor = 1.8
	A = np.zeros([robots,robots]) # Adjacency matrix
	for r in range(0,robots):
		p_rel = p[r,:]-p
		d = (np.sqrt(p_rel[:,0]**2+p_rel[:,1]**2) < rangesensor)
		A[r,:] = d
	G = nx.from_numpy_array(A)
	# f = nx.components.number_connected_components(G)
	largest_cc = max(nx.connected_components(G), key=len)
	f = len(largest_cc)
	return f/robots

def mean_number_of_neighbors(log):
	id_column = 1
	robots = int(log[:,id_column].max())
	p = log[0:robots,(2,3)] # Positions
	rangesensor = 1.8
	A = np.zeros([robots,robots]) # Adjacency matrix
	f = 0
	for r in range(0,robots):
		p_rel = p[r,:]-p
		d = np.where((np.sqrt(p_rel[:,0]**2+p_rel[:,1]**2) < rangesensor))[0]
		n_neighbors = d.size
		f += (n_neighbors-1) / robots
	return f

def expl(log):
	return largest_cluster(log)