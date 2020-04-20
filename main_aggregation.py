#!/usr/bin/env python3
"""
Simulate the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""

import graphless_optimization as opt
from tools import fileHandler as fh
from tools import matrixOperations as matop
import numpy as np
import matplotlib.pyplot as plt
import subprocess, sys
import datetime

## Options
rerun = True
evaluate = True
n = 30 # Robots

###### Simulate ######
from simulator import swarmulator
save_id = "data/" + str(datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S"))
folder = "../swarmulator"
data_folder = folder + "/logs/"
sim = swarmulator.swarmulator(folder) # Initialize
sim.runtime_setting("time_limit", "2000")

if rerun:
	sim.runtime_setting("simulation_realtimefactor", "50")
	sim.runtime_setting("environment", "square")
	sim.make(clean=True,animation=False,logger=False,verbose=True) # Build (if already built, you can skip this)
	# sim.runtime_setting("policy", "") # Use random policy
	subprocess.call("cd " + data_folder + " && rm *.csv", shell=True)
	f = sim.run(n) # Run it, and receive the fitness.

print("done")
###### Optimize ######
H = fh.read_matrix(data_folder,"H")
A = fh.read_matrix(data_folder,"A")
E = fh.read_matrix(data_folder,"E")
des = fh.read_matrix(data_folder,"des")
#np.savez(save_id+"_estimated_data", des=des, H=H, A=A, E=E)
policy = np.ones([A.shape[1],int(A.max())]) / A.shape[1]
result, policy, empty_states = opt.main(policy, des, H, A, E)
np.savez(save_id+"_optimization", des=des, policy=policy, H=H, A=A, E=E)
#data = np.load('data/2020_04_16_10:58:24_optimization.npz')
#H = data['H'].astype(float)
#E = data['E'].astype(float)
#A = data['A'].astype(float)
#policy = np.ones([A.shape[1],int(A.max())]) /2 
#des = data['des'].astype(float)
#prl = np.sum(H,axis=1)+np.sum(E,axis=1)
#Hprl = matop.normalize_rows(H/prl)
#Eprl = matop.normalize_rows(E/prl)
#desprl = matop.normalize_rows(des/prl,axis=0)
#result, policy, empty_states = opt.main(policy,des,Hprl,A,Eprl)

###### Print results to terminal ######
print(H)
print(E)
print(A)
print("States desireability: ", str(des))
print("Unknown states:" + str(empty_states))
print('{:=^40}'.format(' Optimization '))
print("Final fitness: " + str(result.fun))
print("[ policy ]")
np.set_printoptions(threshold=sys.maxsize)
print(policy)

###### Evaluate ######
if evaluate:
	runs = 100
	sim.runtime_setting("time_limit", "1000")
	sim.runtime_setting("simulation_realtimefactor", "50")
	sim.runtime_setting("environment", "square")

	# Benchmark #
	f_0 = []
	sim.runtime_setting("policy", "") # Use random policy
	for i in range(0,runs):
		print('{:=^40}'.format(' Simulator run '))
		print("Run " + str(i) + "/" + str(runs))
		f_0 = np.append(f_0,sim.run(n))

	# Optimized #
	f_n = []
	policy_file = sim.path + "/conf/state_action_matrices/evolved_policy.txt"
	fh.save_to_txt(policy.T, policy_file)
	sim.runtime_setting("policy", policy_file) # Use random policy
	for i in range(0,runs):
		print('{:=^40}'.format(' Simulator run '))
		print("Run " + str(i) + "/"  +str(runs))
		f_n = np.append(f_n,sim.run(n))

	np.savez(save_id+"_validation", f_0=f_0, f_n=f_n)

# Note: a separate script is available to plot the results as histograms
