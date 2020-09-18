#!/usr/bin/env python3
"""
This file is used to generate benchmark performances featuring random policies.
The generated data is used for the comparison in Figure 6 of the paper.

@author: Mario Coppola, 2020
"""

import pickle, argparse
import numpy as np
import parameters
from classes import simulator
from tools import matrixOperations as matop
from tools import fileHandler as fh

# Input argument parser
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('controller', type=str, 
	help="(str) Controller to use during evaluation")
parser.add_argument('-t', type=int, default=200,
	help="(int) Simulation time during benchmark, default = 200s")
parser.add_argument('-n', type=int, default=30, 
	help="(int) Size of swarm, default = 30")
parser.add_argument('-runs', type=int, default=100,
	help="(int) Evaluation runs per policy, default = 100")
parser.add_argument('-iterations', type=int, default=100,
	help="(int) Evaluated random policies, default = 100")
args = parser.parse_args()

# Simulation parameters
fitness, controller, agent, pr_states, pr_actions = parameters.get(args.controller)

# Load and build simulator
sim = simulator.simulator()
sim.make(controller, agent, clean=True, animation=False, logger=False, verbose=False)

# Run it
f = []
for j in range(args.iterations):
	print("----------------------- %i ----------------------"%j)
	# Generate a random policy
	policy = np.random.rand(pr_states,pr_actions)
	policy = np.reshape(policy,(policy.size//pr_actions,pr_actions)) # Resize pol
	if pr_actions > 1: policy = matop.normalize_rows(policy) # Normalize rows

	# Benchmark its performance
	f.append(sim.benchmark(controller, agent, policy, fitness, 
		robots=args.n, runs=args.runs, time_limit=args.t, make=False))

fh.save_pkl(f,"data/%s/benchmark_random_%s_t%i_r%i_runs%i.pkl"%(controller,controller,args.t,args.n,args.runs))