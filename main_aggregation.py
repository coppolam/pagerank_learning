#!/usr/bin/env python3
"""
Simulate the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""
rerun = True

import argparse
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx 
from tools import fitness_functions as f

import aggregation as env
sim = env.aggregation()
tlim = 50
r = 10
inc = 5
if rerun:
	sim.make()
	for i in range(1,inc+1):
		sim.run(run_id=1, time_limit=tlim, robots=r*i, environment="square")
		filename_ext = ("_t%i_r%i" % (tlim, r*i))
		sim.save_learning_data(filename_ext=filename_ext)
else:
	sim.load(sys.argv)
	# sim.sim.plot_log()

sim.disp()
# sim.optimize()

## Re-evaluating
def reevaluate(*args):
	id_column = 1
	robots = int(sim.log[:,id_column].max())
	print(robots)
	time_column = 0
	t = np.unique(sim.log[:,0])
	f_official = np.zeros(t.shape)
	fitness = np.zeros([t.size,len(args)])
	arguments = locals()
	print("Re-evaluating")
	a = 0
	states = np.zeros([t.size,robots])
	des = np.zeros([t.size,sim.H.shape[0]])
	for step in t:
		d = sim.log[np.where(sim.log[:,time_column] == step)]
		fref = 0
		for i in args:
			fitness[a,fref] = i(d)
			fref += 1
		f_official[a] = d[:,5].astype(float).mean()
		states[a] = d[0:robots,4].astype(int)
		for r in np.arange(0,np.max(states[a])+1).astype(int):
			if r < sim.H.shape[0]: # Guard for max state in case inconsistent with Swarmulator
				des[a,r] = np.count_nonzero(states[a] == r)
		a += 1
	return t, f_official, fitness, des

## Fitnesses
def plot_fitness(t,fitness):
	for a in range(fitness.shape[1]):
		plt.plot(t,fitness[:,a]/np.mean(fitness[:,a]))
	plt.ylabel("Fitness")
	plt.xlabel("Time [s]")
	plt.show()

## Correlation
def plot_correlation(fitness):
	for a in range(1,fitness.shape[1]):
		plt.plot(fitness[:,0],fitness[:,a],'*')
		c = np.corrcoef(fitness[:,0],fitness[:,a])[0,1]
		print("Cov 0:", str(a), " = ", str(c))
	plt.ylabel("Local Fitness")
	plt.xlabel("Global Fitness")
	plt.show()

# print("Revaluating fitness")
# t, f_official, fitness, des = reevaluate(
# 	f.expl, 
# 	f.mean_number_of_neighbors) # f.mean_distance_to_rest)
# np.savez(sim.save_id+"_fitness_cc", f_official=f_official, fitness=fitness, des=des)
# print("Saved")

# data1 = np.load("data/1_fitness_cc.npz")
# data2 = np.load("data/2_fitness_cc.npz")
# data3 = np.load("data/3_fitness_cc.npz") # 20 robots
# data4 = np.load("data/4_fitness_cc.npz") # 10 robots
# fitness1 = data1['fitness'].astype(float)
# fitness2 = data2['fitness'].astype(float)
# fitness3 = data3['fitness'].astype(float)
# fitness4 = data4['fitness'].astype(float)
# des1 = data1['des'].astype(float)
# des2 = data2['des'].astype(float)
# des3 = data3['des'].astype(float)
# des4 = data4['des'].astype(float)
# fitness1, ind1 = np.unique(fitness1,axis=0,return_index=True)
# fitness2, ind2 = np.unique(fitness2,axis=0,return_index=True)
# fitness3, ind3 = np.unique(fitness3,axis=0,return_index=True)
# fitness4, ind4 = np.unique(fitness4,axis=0,return_index=True)
# des1 = des1[ind1]
# des2 = des2[ind2]
# des3 = des3[ind3]
# des4 = des4[ind4]
# plt.plot(fitness1[:,0]*10,fitness1[:,1],'*')
# plt.plot(fitness2[:,0]*20,fitness2[:,1],'.')
# plt.plot(fitness3[:,0]*30,fitness3[:,1],'+')
# plt.plot(fitness4[:,0]*40,fitness4[:,1],'.')
# plt.show()

# plot_fitness(t, fitness)
# plot_correlation(fitness)