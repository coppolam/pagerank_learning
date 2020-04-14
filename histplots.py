#!/usr/bin/env python3
"""
Compare the results
@author: Mario Coppola, 2020
"""

import numpy as np
from tools import fileHandler as fh
import matplotlib
import matplotlib.pyplot as plt

#### Input
folder = "data/"
file  = "2020_04_13_18:07:16"
alpha = 0.5;
matplotlib.rcParams['text.usetex'] = False # True for latex style
plot = False
#### Load validation data
data_validation = np.load(folder+file+"_validation.npz")
fitness_0 = data_validation['arr_0'].astype(float)
fitness_n = data_validation['arr_1'].astype(float)

#### Plot validation data
if plot:
    plt.hist(fitness_0, alpha=alpha, density=True, label='$\pi_0$')  # arguments are passed to np.histogram
    plt.hist(fitness_n, alpha=alpha, density=True, label='$\pi_n$')  # arguments are passed to np.histogram
    plt.xlabel("Fitness")
    plt.ylabel("Instances")
    plt.legend(loc='upper right')
    plt.show()

#### Load optimization data
data_policy = np.load(folder+file+"_optimization.npz")
des = data_policy['arr_0'].astype(float)
policy = data_policy['arr_2'].astype(float)
H = data_policy['arr_3'].astype(float)
A = data_policy['arr_4'].astype(float)
E = data_policy['arr_5'].astype(float)

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
