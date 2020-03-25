#!/usr/bin/env python3
"""
Simulate the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""

import graphless_optimization as opt
from tools import fileHandler as fh
from tools import matrixOperations as matop
import numpy as np
import subprocess

###### Simulate ######
from simulator import swarmulator
rerun = False
n = 30 # Robots
folder = "../swarmulator"
data_folder = folder + "/mat/"

if rerun:
    subprocess.call("cd " + data_folder + " && rm *.csv", shell=True)
    s = swarmulator.swarmulator(folder) # Initialize
    s.make(clean=False,animation=True) # Build (if already built, you can skip this)
    f = s.run(n) # Run it, and receive the fitness.

###### Optimize ######
H = fh.read_matrix(data_folder,"H")
A = fh.read_matrix(data_folder,"A")
E = fh.read_matrix(data_folder,"E")
des = fh.read_matrix(data_folder,"des")
policy = 0.5 * np.ones([A.shape[1],1])
result, policy, empty_states = opt.main(policy, des, H, A, E)
fh.save_data(folder + "optimization", des, result, policy)

print("States desireability: ", str(des))
print('{:=^40}'.format(' Optimization '))
print("Final fitness: " + str(result.fun))
print("[ policy ]")
print(policy)
