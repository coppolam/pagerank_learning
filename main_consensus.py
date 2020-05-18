#!/usr/bin/env python3
"""
Simulate the consensus and optimize the behavior
@author: Mario Coppola, 2020
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
matplotlib.rc('text', usetex=True)

from simulator import consensus as sim
from tools import matrixOperations as matop
import graphless_optimization as opt

folder = sys.argv[1]

###### Simulate ######
n = 10 # Number of robots
m = 2 # Choices
sim = sim.consensus_simulator(n=n,m=m,d=0.2)

def simulate():
    print('{:=^40}'.format(' Simulate '))
    runs = 1
    for i in range(0,runs):
        sim.reset(n)
        # sim.e.set_size(sim.e.H.size[1]) # Uncomment to reset estimator between trials
        steps = sim.run()
    np.savez(folder + "sim_n%i_m%i"%(n,m), H=sim.e.H, A=sim.e.A, E=sim.e.E)

###### Optimize ######
def optimize():
    print('{:=^40}'.format(' Optimize '))
    des_idx = np.unique(sim.e.des)
    des = np.zeros([1,sim.policy.shape[0]])[0]
    des[des_idx] = 1
    result, policy, empty_states = opt.main(sim.policy, des, sim.e.H, sim.e.A, sim.e.E)
    np.savez(folder + "optimization_n%i_m%i"%(n,m), perms=sim.perms, des=sim.e.des, result=result, policy=policy)

    print("Final fitness: " + str(result.fun))
    print("[ states | policy ]")
    matop.pretty_print(np.concatenate((sim.perms,policy),axis=1))
    print("Desired states found:", str(des_idx))
    print("Unknown states:" + str(empty_states))
    return policy

###### Validate ######
def validate(policy):
    print('{:=^40}'.format(' Validate '))
    runs = 100
    f = []
    f_n = []
    print('{:=^10}'.format(' Original '))
    for i in tqdm(range(0,runs)):
        sim.reset(n)
        f = np.append(f,sim.run())

    print('{:=^10}'.format(' Optimized '))
    for i in tqdm(range(0,runs)):
        sim.reset(n)
        f_n = np.append(f_n,sim.run(policy=policy))

    np.savez(folder+"validation_n%i_m%i"%(n,m), f=f, f_n=f_n)

simulate()
policy = optimize()
validate(policy)
data = np.load(folder+"validation_n%i_m%i.npz"%(n,m))
f = data["f"].astype(float)
f_n = data["f_n"].astype(float)
_ = plt.hist(f, alpha=0.5, bins='auto', label='Original')  # arguments are passed to np.histogram
_ = plt.hist(f_n, alpha=0.5, bins='auto', label='Optimized')  # arguments are passed to np.histogram
_ = plt.title("Results")
_ = plt.legend(loc='upper right')
_ = plt.savefig(folder+"benchmark.pdf")
