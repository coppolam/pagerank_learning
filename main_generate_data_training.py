#!/usr/bin/env python3
"""
Loop the simulator and gather data on the behavior
@author: Mario Coppola, 2020
"""

import os, argparse
import numpy as np
from tqdm import tqdm
from classes import simulator
from tools import fileHandler as fh
from tools import matrixOperations as matop
import parameters

def save_policy(sim,policy):
	policy_filename = "conf/state_action_matrices/loop_policy_loop.txt"
	policy_file = sim.sim.path + "/" + policy_filename
	if policy.shape[1] == 1: fh.save_to_txt(policy.T, policy_file) # Number of columns = 1
	else: fh.save_to_txt(policy, policy_file)
	return policy_filename

# Input argument parser
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('controller', type=str, help="(str) Controller to use during evaluation")
parser.add_argument('-t', type=int, default=200, help="(int) Simulation time during benchmark, default = 200s")
parser.add_argument('-n', type=int, default=30, help="(int) Max size of swarm, default = 30")
parser.add_argument('-id', type=int, default=np.random.randint(1000), help="(int) ID of run, default = random")
parser.add_argument('-iterations', type=int, default=500, help="(int) Number of iterations")
parser.add_argument('-environment', type=str, default="square20", help="(str) Controller to use during evaluation")
parser.add_argument('-animate', action='store_true', help="(bool) Animate flag to true")
args = parser.parse_args()

# Simulation parameters
fitness, controller, agent, pr_states, pr_actions = parameters.get(args.controller)

# Load and build
sim = simulator.simulator(savefolder="data/%s/data_%i/"%(controller,args.id))
sim.make(controller, agent, animation=args.animate, verbose=False)

for i in tqdm(range(args.iterations)):
	# Random policy
	policy = np.random.rand(pr_states,pr_actions)
	policy = np.reshape(policy,(policy.size//pr_actions,pr_actions)) # Resize pol
	if pr_actions > 1: policy = matop.normalize_rows(policy) # Normalize rows
	policy_filename = save_policy(sim, policy)

	# Run
	sim.run(time_limit=args.t, robots=np.random.randint(1,args.n), 
		environment=args.environment, policy=policy_filename, 
		pr_states=pr_states, pr_actions=pr_actions, run_id=args.id, fitness=fitness)

	# Save
	filename_ext = "%s_t%i_r%i_id%s_%i" % (args.controller, args.t, args.n, sim.run_id, i)
	learning_file = sim.save_learning_data(filename_ext=filename_ext)