#!/usr/bin/env python3
"""
Simulate the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""

import graphless_optimization as opt
from tools import fileHandler as fh
from tools import matrixOperations as matop
import numpy as np
import subprocess, sys

###### Simulate ######
from simulator import swarmulator
rerun = True
n = 30 # Robots
folder = "../swarmulator"
data_folder = folder + "/logs/"

if rerun:
    subprocess.call("cd " + data_folder + " && rm *.csv", shell=True)
    s = swarmulator.swarmulator(folder) # Initialize
    s.make(clean=True,animation=True,logger=True) # Build (if already built, you can skip this)
    f = s.run(n) # Run it, and receive the fitness.

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

NEWLINE_SIZE_IN_BYTES = -1  # -2 on Windows?
with open('policy.txt', 'wb') as fout:  # Note 'wb' instead of 'w'
    np.savetxt(fout, policy, delimiter=" ", fmt='%.3f')
    fout.seek(NEWLINE_SIZE_IN_BYTES, 2)
    fout.truncate()
