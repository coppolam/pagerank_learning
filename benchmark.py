#!/usr/bin/env python3
"""
Simulate the aggregation and optimize the behavior
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
parser.add_argument('controller', type=str, help="(str) Controller to use during evaluation")
parser.add_argument('-t', type=int, help="(int) Simulation time during benchmark, default = 500s", default=500)
parser.add_argument('-n', type=int, help="(int) Size of swarm, default = 20", default=20)
parser.add_argument('-runs', type=int, help="(int) Evaluation runs, default = 20", default=20)
parser.add_argument('-iterations', type=int, help="(int) Evaluation runs, default = 100", default=100)
args = parser.parse_args()

# Simulation parameters
fitness, controller, agent, pr_states, pr_actions = parameters.get(args.controller)

## Step 2: PageRank optimize
sim = simulator.simulator()
sim.make(controller, agent, clean=True, animation=False, logger=False, verbose=False)

f = []
for j in range(args.iterations):
	print("----------------------- %i ----------------------"%j)
	policy = np.random.rand(pr_states,pr_actions)
	policy = np.reshape(policy,(policy.size//pr_actions,pr_actions)) # Resize pol
	if pr_actions > 1: policy = matop.normalize_rows(policy) # Normalize rows

	f.append(sim.benchmark(controller, agent, policy, fitness, 
		robots=args.n, runs=args.runs, time_limit=args.t, make=False))

fh.save_pkl(f,"data/%s/benchmark_random_%s.pkl"%(controller,controller))