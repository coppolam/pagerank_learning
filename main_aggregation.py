#!/usr/bin/env python3
"""
Simulate the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""

import aggregation, sys
import numpy as np
rerun = False
import matplotlib
import matplotlib.pyplot as plt
import networkx 

sim = aggregation.aggregation()

if rerun:
	sim.run(time_limit=100)
	sim.save()
else:
	sim.load(sys.argv)
	# sim.sim.plot_log()

def evaluate_fitness(data):
	id_column = 1
	robots = int(data[:,id_column].max())
	p = data[0:robots,(2,3)] # Positions
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

time_column = 0
t = np.unique(sim.data[:,0])
fitness = np.zeros(t.shape)
a = 0
for step in t:
	d = sim.data[np.where(sim.data[:,time_column] == step)]
	fitness[a] = evaluate_fitness(d)
	a += 1

plt.plot(np.arange(0,fitness.size),fitness)
plt.show()
# sim.optimize()
# sim.evaluate(runs=1)
# sim.histplots()