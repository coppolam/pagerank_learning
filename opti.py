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
	policy_filename = "conf/state_action_matrices/aggregation_policy_loop.txt"
	policy_file = sim.sim.path + "/" + policy_filename
	if policy.shape[1] == 1: fh.save_to_txt(policy.T, policy_file) # Number of columns = 1
	else: fh.save_to_txt(policy, policy_file)
	return policy_filename

# Input argument parser
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('folder', type=str, help="(str) Model file")
parser.add_argument('controller', type=str, help="(str) Controller to use during evaluation")
parser.add_argument('-t', type=int, help="(int) Simulation time during benchmark, default = 500s", default=500)
parser.add_argument('-n', type=int, help="(int) Size of swarm, default = 30", default=30)
parser.add_argument('-runs', type=int, help="(int) Evaluation runs, default = 100", default=100)
parser.add_argument('-id', type=int, help="(int) ID of run, default = 1", default=1)
parser.add_argument('-animate', type=bool, help="(bool) If True, does not do a benchmark but only shows a swarm with the optimized controller, default = False", default=False)
parser.add_argument('-iterations', type=int, help="(int) Number of iterations", default=20)
parser.add_argument('-log', type=int, help="(int) If set, logs one run for the indicated amount of time, default = None", default=None)
args = parser.parse_args()

# Default values for each controller
if args.controller == "aggregation":
	fitness = "aggregation_clusters"
	agent = "particle"
	policy = np.ones((8,1))/2 # all = 1/2
	pr_states = 8
	pr_actions = 1
elif args.controller == "dispersion":
	args.controller = "aggregation"
	agent = "particle"
	fitness = "dispersion_clusters"
	policy = np.ones((8,1))/2 # all = 1/2
	pr_states = 8
	pr_actions = 1
elif args.controller == "pfsm_exploration":
	fitness = "aggregation_clusters"
	agent = "particle_oriented"
	policy = np.ones((16,8))/8 # all = 1/8
	pr_states = 16
	pr_actions = 8
elif args.controller == "pfsm_dispersion":
	args.controller = "pfsm_exploration"
	agent = "particle_oriented"
	fitness = "dispersion_clusters"
	policy = np.ones((16,8))/8 # all = 1/8
	pr_states = 16
	pr_actions = 8
elif args.controller == "pfsm_exploration_mod":
	fitness = "aggregation_clusters"
	agent = "particle_oriented"
	policy = np.ones((16,8))/8 # all 1/8
	pr_states = 16
	pr_actions = 8
elif args.controller == "forage":
	fitness = "food"
	agent = "particle_oriented"
	policy = np.ones((30,1))/2 # all = 1/2
	pr_states = 30
	pr_actions = 1
else:
	ValueError("Unknown controller!")

sim = simulator.simulator(savefolder=args.folder)
des_nn = desired_states_extractor.desired_states_extractor()
i = 0
for filename in os.listdir(args.folder):
	if filename.endswith(".npz"):
		sim.load(args.folder+filename) if i == 0 else sim.load_update(args.folder+filename)
		des_nn.train(args.folder+filename,load=False,verbose=True)
		i += 1

des = des_nn.get_des()
policy = sim.optimize(policy, des)
print("Final")
print(policy)
sim.make(args.controller, agent, animation=True, verbose=False)
policy_filename = save_policy(sim, policy)
sim.run(time_limit=0, robots=args.n, environment="square", policy=policy_filename, 
	pr_states=pr_states, pr_actions=pr_actions, run_id=args.id, fitness=fitness)
