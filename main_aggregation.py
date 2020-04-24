#!/usr/bin/env python3
"""
Simulate the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""

rerun = False

import aggregation, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx 
from tools import fitness_functions as f

sim = aggregation.aggregation()

if rerun:
	sim.run(time_limit=100)
	sim.save()
else:
	sim.load(sys.argv)
	# sim.sim.plot_log()

sim.disp()

# Re-evaluating
def reevaluate(*args):
	id_column = 1
	robots = int(sim.log[:,id_column].max())
	time_column = 0
	t = np.unique(sim.log[:,0])
	f_official = np.zeros(t.shape)
	fitness = np.zeros([t.size,len(args)])
	arguments = locals()
	print("Re-evaluating")
	a = 0
	states = np.zeros([t.size,robots])
	c = np.zeros([t.size,8])
	for step in t:
		d = sim.log[np.where(sim.log[:,time_column] == step)]
		fref = 0
		for i in args:
			fitness[a,fref] = i(d)
			fref += 1
		f_official[a] = d[:,5].astype(float).mean()
		states[a] = d[:,4].astype(int)
		for r in np.arange(0,np.max(states[a])+1).astype(int):
			c[a,r] = np.count_nonzero(states[a] == r)
		a += 1
	return t, f_official, fitness

# Fitnesses
def plot_fitness(t,f_official,fitness):
	plt.plot(t,f_official/np.mean(f_official))
	for a in range(fitness.shape[1]):
		plt.plot(t,fitness[:,a]/np.mean(fitness[:,a]))
	plt.ylabel("Fitness")
	plt.xlabel("Time [s]")
	plt.show()

## Correlation
def plot_correlation(fitness):
	for a in range(1,fitness.shape[1]):
		plt.plot(fitness[:,0],fitness[:,a])
		c = np.corrcoef(fitness[:,0],fitness[:,a])[0,1]
		print("Cov 0:", str(a), " = ", str(c))
	plt.ylabel("Fitness")
	plt.xlabel("Fitness")
	plt.show()

t, f_official, fitness = reevaluate(
	f.number_of_clusters, 
	f.mean_number_of_neighbors,
	f.mean_distance_to_rest)

plot_fitness(t, f_official, fitness)