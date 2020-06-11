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

from simulators import consensus as sim
from tools import matrixOperations as matop
from classes import pagerank_optimization as opt

folder = sys.argv[1]
n = 10 # Number of robots
m = 2 # Choices
sim = sim.consensus_simulator(n=n,m=m,d=0.2)

def simulate():
    print("Simulating and generating data")
    runs = 1
    for i in range(runs):
        sim.reset(n)
        # sim.e.set_size(sim.e.H.size[1]) # Uncomment to reset estimator between trials
        steps = sim.run()
    np.savez(folder + "sim_n%i_m%i"%(n,m), H=sim.e.H, A=sim.e.A, E=sim.e.E)

def optimize():
    print("Optimizing")
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

def validate(policy):
    runs = 100
    f_0 = []
    f_n = []

    print("Evaluating baseline policy")
    for i in tqdm(range(runs)):
        sim.reset(n)
        f_0 = np.append(f_0,sim.run())

    print("Evaluating optimized policy")
    for i in tqdm(range(runs)):
        sim.reset(n)
        f_n = np.append(f_n,sim.run(policy=policy))

    np.savez(folder+"validation_n%i_m%i"%(n,m), f=f_0, f_n=f_n)
    return f_0, f_n

if __name__ == "__main__":
    # Simulate and gather data  
    simulate()

    # Optimize
    policy = optimize()

    # Validate
    f_0, f_n = validate(policy)
    
    # Plot evalauation
    alpha = 0.5
    plt.hist(f_0, alpha=alpha, bins='auto', label='Original')
    plt.hist(f_n, alpha=alpha, bins='auto', label='Optimized')
    plt.title("Results")
    plt.legend(loc='upper right')
    plt.savefig(folder+"benchmark.pdf")