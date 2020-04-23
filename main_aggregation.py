#!/usr/bin/env python3
"""
Simulate the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""

rerun = True

import aggregation, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx 

sim = aggregation.aggregation()

if rerun:
	sim.run(time_limit=10000)
	sim.save()
else:
	sim.load(sys.argv)
	# sim.sim.plot_log()

def evaluate_fitness(log):
	id_column = 1
	robots = int(log[:,id_column].max())
	p = log[0:robots,(2,3)] # Positions
	rangesensor = 1.8
	f = 0

	# Clustering coefficient
	A = np.zeros([robots,robots]) # Adjacency matrix
	for r in range(0,robots):
		p_rel = p[r,:]-p
		d = (np.sqrt(p_rel[:,0]**2+p_rel[:,1]**2) < rangesensor)
		A[r,:] = d
	G = networkx.from_numpy_array(A)
	f = networkx.average_clustering(G)

	# Mean number of neighbors
	# for r in range(0,robots):
	# 	p_rel = p[r,:]-p
	# 	d = np.where((np.sqrt(p_rel[:,0]**2+p_rel[:,1]**2) < rangesensor))[0]
	# 	n_neighbors = d.size
	# 	f += (n_neighbors-1) / robots

	# Mean distance to neighbors
	# for r in range(0,robots):
	# 	p_rel = p[r,:]-p
	# 	d = np.sqrt(p_rel[:,0]**2+p_rel[:,1]**2)
	# 	f += d.mean()/robots
	return f

def analyze():
	id_column = 1
	robots = int(sim.log[:,id_column].max())
	time_column = 0
	t = np.unique(sim.log[:,0])
	fitness = np.zeros(t.shape)
	states = np.zeros([t.size,robots])
	c = np.zeros([t.size,8])
	a = 0
	for step in t:
		d = sim.log[np.where(sim.log[:,time_column] == step)]
		fitness[a] = evaluate_fitness(d)
		states[a] = d[:,4].astype(int)
		for r in np.arange(0,np.max(states[a])+1).astype(int):
			c[a,r] = np.count_nonzero(states[a] == r)
		a += 1

	fig = plt.figure()
	ax = fig.gca()
	ax.plot(fitness,c)
	a = str(range(0,8))
	ax.legend(list(np.arange(0,8).astype(str)))
	ax.set_xlabel("Desirability [-]")
	ax.set_ylabel("Fitness [-]")
	# plt.show()
