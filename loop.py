#!/usr/bin/env python3
"""
Simulate the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""

import pickle, sys, matplotlib, os, argparse
import numpy as np
import matplotlib.pyplot as plt
matplotlib.rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from classes import simulator, evolution, desired_states_extractor
from scipy.special import softmax

def plot_benchmark(file):
	data = np.load(file)
	alpha = 0.5
	plt.figure(figsize=(6,3))
	if "f_0" in data.files: plt.hist(data["f_0"].astype(float), alpha=alpha, label='$\pi_0$')
	if "f_n" in data.files: plt.hist(data["f_n"].astype(float), alpha=alpha, label='$\pi_n$')
	if "f_s" in data.files: plt.hist(data["f_s"].astype(float), alpha=alpha, label='$\pi*$')
	plt.legend()
	plt.xlabel("Fitness [-]")
	plt.ylabel("Frequency")
	plt.gcf().subplots_adjust(bottom=0.15)
	folder = os.path.dirname(file) + "/figures/"
	if not os.path.exists(os.path.dirname(folder)): os.makedirs(os.path.dirname(folder))
	filename_raw = os.path.splitext(os.path.basename(file))[0]
	plt.savefig(folder+"%s.pdf"%filename_raw)
	plt.clf()

# Input argument parser
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('controller', type=str, help="(str) Controller to use during evaluation")
parser.add_argument('agent', type=str, help="(str) Agent to use during evaluation")
parser.add_argument('-t', type=int, help="(int) Simulation time during benchmark, default = 500s", default=500)
parser.add_argument('-n', type=int, help="(int) Size of swarm, default = 30", default=30)
parser.add_argument('-runs', type=int, help="(int) Evaluation runs, default = 100", default=100)
parser.add_argument('-id', type=int, help="(int) ID of run, default = 1", default=1)
parser.add_argument('-animate', type=bool, help="(bool) If True, does not do a benchmark but only shows a swarm with the optimized controller, default = False", default=False)
parser.add_argument('-log', type=int, help="(int) If set, logs one run for the indicated amount of time, default = None", default=None)
args = parser.parse_args()

# Default values for each controller
if args.controller == "aggregation":
	fitness = "aggregation_clusters"
	policy = np.ones((8,1))/2 # all = 1/2
	pr_states = 8
	pr_actions = 1
elif args.controller == "pfsm_exploration":
	fitness = "aggregation_clusters"
	policy = np.ones((16,8))/8 # all = 1/8
	pr_states = 16
	pr_actions = 8
elif args.controller == "pfsm_exploration_mod":
	fitness = "aggregation_clusters"
	policy = np.ones((16,8))/8 # all 1/8
	pr_states = 16
	pr_actions = 8
elif args.controller == "forage":
	fitness = "food"
	policy = np.ones((16,1))/2 # all = 1/2
else:
	ValueError("Unknown controller!")

# Load and build
sim = simulator.simulator(savefolder="data/%s_%s/loop_%i/"%(args.controller,args.agent,args.id)) # Load high level simulator interface, connected to swarmulator
sim.make(args.controller, args.agent, animation=args.animate, verbose=False) # Build
des_nn = desired_states_extractor.desired_states_extractor()

n = 20 # Number of iterations
for i in range(n):
	print("Iteration %i"%i)
	policy = softmax(policy,axis=1)

	# Save policy file to test
	from tools import fileHandler as fh
	policy_filename = "conf/state_action_matrices/aggregation_policy_loop.txt"
	policy_file = sim.sim.path + "/" + policy_filename
	if policy.shape[1] == 1: fh.save_to_txt(policy.T, policy_file) # Number of columns = 1
	else: fh.save_to_txt(policy, policy_file)
	
	# Run
	sim.run(time_limit=args.t, robots=args.n, environment="square", policy=	policy_filename, 
		pr_states=pr_states, pr_actions=pr_actions, run_id=args.id, fitness=fitness)

	# Save data
	filename_ext = ("%s_%s_t%i_r%i_id%s_%i" % (args.controller, args.agent, args.t, args.n, sim.run_id, i))
	learning_file = sim.save_learning_data(filename_ext=filename_ext)

	# Optimization procedure
	## Step 1: Get the desired states
	des = des_nn.run(learning_file+".npz",load=False,verbose=True)

	## Step 2: PageRank optimize
	sim.load(learning_file+".npz") if i == 0 else sim.load_update(learning_file+".npz",i)
	policy = sim.optimize(policy, des)
	print("Optimal Policy")
	print(policy)

# Test
sim.make(args.controller, args.agent, animation=True, verbose=False) # Build

print("Final",i)
print(policy)

# Save policy file to test
from tools import fileHandler as fh
policy_filename = "conf/state_action_matrices/aggregation_policy_loop.txt"
policy_file = sim.sim.path + "/" + policy_filename
if policy.shape[1] == 1: fh.save_to_txt(policy.T, policy_file) # Number of columns = 1
else: fh.save_to_txt(policy, policy_file)

# Observe
sim.run(time_limit=0, robots=args.n, environment="square", policy=policy_filename, 
	pr_states=pr_states, pr_actions=pr_actions, run_id=args.id, fitness=fitness)