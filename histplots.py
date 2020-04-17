#!/usr/bin/env python3
"""
Compare the results
@author: Mario Coppola, 2020
"""

import numpy as np
from tools import fileHandler as fh
import matplotlib
import matplotlib.pyplot as plt
import sys

#### Input
folder = "data/"
file  = "2020_04_17_19:11:43"
alpha = 0.5;
matplotlib.rcParams['text.usetex'] = False # True for latex style
plot = True
#### Load validation data
data_validation = np.load(folder+file+"_validation.npz")
fitness_0 = data_validation['f_0'].astype(float)
fitness_n = data_validation['f_n'].astype(float)
#data_validation_05 = np.load(folder+"2020_04_16_11:56:28_validation_0.5.npz")
#data_validation_bt = np.load(folder+"2020_04_16_14:15:29_validation_behaviortree.npz")
#fitness_n2 = data_validation_05['f_n'].astype(float)
#fitness_n3 = data_validation_bt['f_n'].astype(float)
#### Plot validation data
if plot:
    plt.hist(fitness_0, alpha=alpha, density=True, label='$\pi_0$')  # arguments are passed to np.histogram
    plt.hist(fitness_n, alpha=alpha, density=True, label='$\pi_n$')  # arguments are passed to np.histogram
#    plt.hist(fitness_n2,alpha=alpha, density=True, label='$\pi_n0.5$')
#    plt.hist(fitness_n3,alpha=alpha, density=True, label='$\pi_nbt$')
    plt.xlabel("Fitness")
    plt.ylabel("Instances")
    plt.legend(loc='upper right')
    plt.show()

#### Load optimization data
np.set_printoptions(threshold=sys.maxsize,suppress=True)
data_policy = np.load(folder+file+"_optimization.npz")
des = data_policy['des'].astype(float)
policy = data_policy['policy'].astype(float)
H = data_policy['H'].astype(float)
A = data_policy['A'].astype(float)
E = data_policy['E'].astype(float)

print("H = ")
print(H)
print("E = ")
print(E)
print("A = ")
print(A)
print("des = ")
print(des)
print("policy = ")
print(policy)

#### Investigate policy behavior
###### Simulate ######
print('{:=^40}'.format(' Simulator run '))
from simulator import swarmulator
rerun = True
n = 30 # Robots
folder = "../swarmulator"
sim = swarmulator.swarmulator(folder) # Initialize
sim.runtime_setting("simulation_realtimefactor", "50")
sim.runtime_setting("time_limit", "0")
sim.runtime_setting("environment", "square")
sim.make(clean=True,animation=True,logger=False) # Build (if already built, you can skip this)

policy_file = sim.path + "/conf/state_action_matrices/aggregation_policy_test.txt"
fh.save_to_txt(policy.T, policy_file)
sim.runtime_setting("policy", policy_file) # Use random policy
f = sim.run(n) # Run it, and receive the fitness.
