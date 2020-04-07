#!/usr/bin/env python3
"""
Simulate the consensus and optimize the behavior
@author: Mario Coppola, 2020
"""

from simulator import consensus as sim
from tools import fileHandler as fh
from tools import matrixOperations as matop
import graphless_optimization as opt
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(3)
folder = fh.make_folder("data")

###### Simulate ######
n = 10 # Number of robots
m = 2 # Choices
runs = 1
sim = sim.consensus_simulator(n=n,m=m,d=0.2)
for i in range(0,runs):
    print('{:=^40}'.format(' Simulator run '))
    sim.reset(n)
    # sim.e.set_size(sim.e.H.size[1]) # Uncomment to reset estimator between trials
    steps = sim.run()
    fh.save_data(folder + "sim" + str(i), sim.e.H, sim.e.A, sim.e.E, steps)

###### Optimize ######
des_idx = np.unique(sim.e.des)
des = np.zeros([1,sim.policy.shape[0]])[0]
des[des_idx] = 1
result, policy, empty_states = opt.main(sim.policy, des, sim.e.H, sim.e.A, sim.e.E)
fh.save_data(folder + "optimization", sim.perms, sim.e.des, result, policy)

print('{:=^40}'.format(' Optimization '))
print("Final fitness: " + str(result.fun))
print("[ states | policy ]")
matop.pretty_print(np.concatenate((sim.perms,policy),axis=1))
print("Desired states found:", str(des_idx))
print("Unknown states:" + str(empty_states))

###### Validate ######
runs = 500
steps = []
steps_n = []
for i in range(0,runs):
    print('{:=^40}'.format(' Simulator run '))
    sim.reset(n)
    steps = np.append(steps,sim.run())

for i in range(0,runs):
    print('{:=^40}'.format(' Simulator run '))
    sim.reset(n)
    steps_n = np.append(steps_n,sim.run(policy=policy))

_ = plt.hist(list(steps), alpha=0.5, bins='auto', label='original')  # arguments are passed to np.histogram
_ = plt.hist(list(steps_n), alpha=0.5, bins='auto', label='new')  # arguments are passed to np.histogram
plt.title("Results")
plt.legend(loc='upper right')
plt.show()
