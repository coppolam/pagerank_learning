#!/usr/bin/env python3
"""
Loop the aggregation and optimize the behavior
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
from tools import fileHandler as fh
from tools import matrixOperations as matop
import parameters

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

def save_policy(sim,policy):
	policy_filename = "conf/state_action_matrices/aggregation_policy_learnloop.txt"
	policy_file = sim.sim.path + "/" + policy_filename
	if policy.shape[1] == 1: fh.save_to_txt(policy.T, policy_file) # Number of columns = 1
	else: fh.save_to_txt(policy, policy_file)
	return policy_filename

# Input argument parser
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('controller', type=str, help="(str) Controller to use during evaluation")
parser.add_argument('folder_training', type=str, help="(str) Controller to use during evaluation")
parser.add_argument('-t', type=int, help="(int) Simulation time during benchmark, default = 500s", default=500)
parser.add_argument('-n', type=int, help="(int) Size of swarm, default = 30", default=20)
parser.add_argument('-runs', type=int, help="(int) Evaluation runs, default = 100", default=20)
parser.add_argument('-id', type=int, help="(int) ID of run, default = 1", default=1)
parser.add_argument('-animate', type=bool, help="(bool) If True, does not do a benchmark but only shows a swarm with the optimized controller, default = False", default=False)
parser.add_argument('-iterations', type=int, help="(int) Number of iterations", default=1)
parser.add_argument('-log', type=int, help="(int) If set, logs one run for the indicated amount of time, default = None", default=None)
args = parser.parse_args()

# Simulation parameters
fitness, controller, agent, pr_states, pr_actions = parameters.get(args.controller)

# Load and build
sim = simulator.simulator(savefolder="data/%s/learnloop_%i/"%(controller,args.id)) # Load high level simulator interface, connected to swarmulator
sim.make(controller, agent, animation=args.animate, verbose=False) # Build

# Desired states
dse = desired_states_extractor.desired_states_extractor()
dse.load_model("data/%s/models.pkl"%controller)
des = dse.get_des(dim=pr_states)
print(des)

# Initialize policy
policy = np.random.rand(pr_states,pr_actions)
filelist_training = [f for f in os.listdir(args.folder_training) if f.endswith('.npz')]

v = []
for j, filename in enumerate(sorted(filelist_training)):
	if j == 0: sim.load(args.folder_training+filename,verbose=False)
	else: sim.load_update(args.folder_training+filename)

for i in range(args.iterations):
	policy = np.reshape(policy,(policy.size//pr_actions,pr_actions)) # Resize pol
	if pr_actions > 1: policy = matop.normalize_rows(policy) # Normalize rows
	policy_filename = save_policy(sim, policy)
	print("Iteration %i"%i)
	print(policy)

	# Run
	sim.run(time_limit=args.t, robots=args.n, environment="square", policy=	policy_filename, 
		pr_states=pr_states, pr_actions=pr_actions, run_id=args.id, fitness=fitness)

	# Save data
	filename_ext = "%s_%s_t%i_r%i_id%s_%i" % (controller, agent, args.t, args.n, sim.run_id, i)
	learning_file = sim.save_learning_data(filename_ext=filename_ext)

	# Optimization procedure
	## Step 1: Update the desired states with new sim data
	des = dse.run(learning_file+".npz", load=False, verbose=True)

	## Step 2: PageRank optimize
	# sim.load(learning_file+".npz") if i == 0 else 
	sim.load_update(learning_file+".npz",discount=1.0)
	policy = sim.optimize(policy, des)
	print("Optimal policy")

print(policy)
f = sim.benchmark(controller, agent, policy, fitness, robots=args.n, runs=args.runs, time_limit=args.t, make=True)

import plots_paper_benchmark as b
b.benchmark("data/forage/benchmark_random_forage.npz",new=f)