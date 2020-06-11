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

if __name__ == "__main__":
	# Input argument parser
	parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
	parser.add_argument('file', type=str, help="(str) Relative path to simulation file to use")
	parser.add_argument('controller', type=str, help="(str) Controller to use during evaluation")
	parser.add_argument('agent', type=str, help="(str) Agent to use during evaluation")
	parser.add_argument('-t', type=int, help="(int) Simulation time during benchmark, default = 100s", default=100)
	parser.add_argument('-n', type=int, help="(int) Size of swarm, default = 30", default=30)
	parser.add_argument('-runs', type=int, help="(int) Evaluation runs, default = 100", default=100)
	parser.add_argument('-id', type=int, help="(int) ID of run, default = 1", default=1)
	parser.add_argument('-observe', type=bool, help="(bool) If True, does not do a benchmark but only shows a swarm with the optimized controller, default = False", default=False)
	args = parser.parse_args()
	folder = os.path.dirname(args.file)
	filename_raw = os.path.splitext(os.path.basename(args.file))[0]

	# Default values for each controller
	if args.controller == "aggregation":
		fitness = "aggregation_clusters"
		p_0 = np.ones((8,1))/2 # all = 1/2
	elif args.controller == "pfsm_exploration":
		fitness = "aggregation_clusters"
		p_0 = np.ones((16,8))/8 # all = 1/8
	elif args.controller == "pfsm_exploration_mod":
		fitness = "aggregation_clusters"
		p_0 = np.ones((16,8))/8 # all 1/8
	elif args.controller == "forage":
		fitness = "food"
		p_0 = np.ones((16,1))/2 # all = 1/2
	else:
		ValueError("Unknown controller!")

	# Optimization procedure
	## Step 1: Get the desired states
	des = desired_states_extractor.desired_states_extractor().run(args.file,verbose=True)

	## Step 2: PageRank optimize
	sim = simulator.simulator()
	sim.load(args.file)
	# sim.disp()
	p_n =  sim.optimize(p_0,des)

	# Benchmark, either fully (if observe = False) or visually (if observe = True)
	if args.observe:
    	# Just do one run and observe what happens visually
		sim.observe(args.controller, args.agent, p_n, fitness, robots=args.n)
	else: 
		# Run a proper benchmark against unoptimized
		f_0 = sim.benchmark(args.controller, args.agent, p_0, fitness, robots=args.n, runs=args.runs, time_limit=args.t)
		f_n = sim.benchmark(args.controller, args.agent, p_n, fitness, robots=args.n, runs=args.runs, time_limit=args.t)

		# Save evaluation data
		data = np.savez(folder + "/benchmark_%s.npz"%filename_raw, f_0=f_0, f_n=f_n, p_0=p_0, p_n=p_n)
		
		# Plot evaluation data
		plot_benchmark(folder + "/benchmark_%s.npz"%filename_raw)
		print("Done")