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
import subprocess, sys, shutil, os

def save_to_txt(mat,name):
	NEWLINE_SIZE_IN_BYTES = -1  # -2 on Windows?
	with open(name, 'wb') as fout:  # Note 'wb' instead of 'w'
		np.savetxt(fout, mat, delimiter=" ", fmt='%.3f')
		fout.seek(NEWLINE_SIZE_IN_BYTES, 2)
		fout.truncate()

###### Simulate ######
from simulator import swarmulator
rerun = False
n = 30 # Robots
folder = "../swarmulator"
data_folder = folder + "/logs/"
# subprocess.call("cd " + data_folder + " && rm *.csv", shell=True)
sim = swarmulator.swarmulator(folder) # Initialize
sim.runtime_setting("simulation_realtimefactor", "50")
sim.runtime_setting("time_limit", "2000")
sim.runtime_setting("environment", "random")
policy = np.ones((255,8)) / 8
policy_file = sim.path + "/conf/state_action_matrices/policy_random.txt"
save_to_txt(policy, policy_file)
sim.runtime_setting("policy","./conf/state_action_matrices/policy_random.txt")
 
if rerun:
	sim.make(clean=True,animation=True,logger=False) # Build (if already built, you can skip this)
	f = sim.run(n) # Run it, and receive the fitness.

###### Optimize ######
H = fh.read_matrix(data_folder,"H")
A = fh.read_matrix(data_folder,"A")
E = fh.read_matrix(data_folder,"E")
des = fh.read_matrix(data_folder,"des")
policy = np.ones([A.shape[1],int(A.max())]) / A.max()
result, policy, empty_states = opt.main(policy, des, H, A, E)
fh.save_data(folder + "optimization", des, result, policy)

print("States desireability: ", str(des))
print("Unknown states:" + str(empty_states))
print('{:=^40}'.format(' Optimization '))
print("Final fitness: " + str(result.fun))
print("[ policy ]")
np.set_printoptions(threshold=sys.maxsize)
print(policy)

###### Validate ######
runs = 100
steps = []
steps_n = []
policy_file = sim.path + "/conf/state_action_matrices/exploration_policy.txt"
save_to_txt(policy.T, policy_file)
sim.runtime_setting("time_limit", "500")

for i in range(0,runs):
	print('{:=^40}'.format(' Simulator run '))
	sim.runtime_setting("policy", "") # Use random policy
	fitness = np.append(steps,sim.run(n))

for i in range(0,runs):
	print('{:=^40}'.format(' Simulator run '))
	sim.runtime_setting("policy", policy_file) # Use random policy
	fitness_n = np.append(steps_n,sim.run(n))

_ = plt.hist(list(fitness), alpha=0.5, bins='auto', label='original')  # arguments are passed to np.histogram
_ = plt.hist(list(fitness_n), alpha=0.5, bins='auto', label='new')  # arguments are passed to np.histogram
plt.title("Results")
plt.legend(loc='upper right')
plt.show()
