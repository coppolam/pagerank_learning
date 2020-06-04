#!/usr/bin/env python3
"""
Simulate the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""

import pickle, sys, matplotlib, os, argparse
import matplotlib.pyplot as plt
matplotlib.rc('text', usetex=True)
import numpy as np
import simulator
import evolution
import desired_states_extractor

sim = simulator.simulator()

def optimize(file,p0,des):
	sim.load(file)
	sim.disp()
	return sim.optimize(p0,des)
	
def plot_benchmark(loadfile):
	## Plot
	data = np.load(loadfile)
	alpha = 0.5
	if "f_0" in data.files: plt.hist(data["f_0"].astype(float), alpha=alpha, label='$\pi_0$')
	if "f_n" in data.files: plt.hist(data["f_n"].astype(float), alpha=alpha, label='$\pi_n$')
	if "f_s" in data.files: plt.hist(data["f_s"].astype(float), alpha=alpha, label='$\pi*$')
	plt.legend()

	## Save to "figures" subfolder with the same name
	folder = os.path.dirname(loadfile) + "figures/"
	directory = os.path.dirname(folder)
	if not os.path.exists(directory): os.makedirs(directory)
	plt.savefig(folder+"%s.pdf"%loadfile)
	plt.clf()

if __name__ == "__main__":
	###########################
	#  Input argument parser  #
	###########################
	parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
	parser.add_argument('file', type=str, help="Simulation file to use")
	parser.add_argument('controller', type=str, help="Controller to use")
	parser.add_argument('agent', type=str, help="Agent to use")
	parser.add_argument('-t', type=int, help="Max time of simulation. Default = 10000s", default=10000)
	parser.add_argument('-n', type=int, help="Size of swarm. Default = 30", default=30)
	parser.add_argument('-runs', type=int, help="Size of swarm. Default = 30", default=100)
	parser.add_argument('-id', type=int, help="ID", default=1)
	parser.add_argument('-observe', type=bool, help="", default=False)
	args = parser.parse_args()

	###########################
	#     Default values      #
	###########################
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

	############################
	#  Optimization procedure  #
	############################
	## Step 1: Get the desired states
	des = desired_states_extractor.desired_states_extractor().run(args.file,verbose=True)

	## Step 2: PageRank optimize
	p_n = optimize(args.file, p_0, des)

	#############
	# Benchmark #
	#############
	if args.observe:
    	# Just do one run and observe what happens visually
		sim.observe(args.controller, args.agent, p_n, robots=args.n)
	else: 
		# Run a proper benchmark against unoptimized
		f_0 = sim.benchmark(args.controller, args.agent, p_0, fitness,robots=args.n,runs=args.runs,time_limit=args.t)
		f_n = sim.benchmark(args.controller, args.agent, p_n, fitness,robots=args.n,runs=args.runs,time_limit=args.t)

		# Save
		folder = os.path.dirname(args.file)
		filename = os.path.basename(args.file)
		data_validation = np.savez(folder + "benchmark_%s"%filename,f_0=f_0,f_n=f_n,p_0=p_0,p_n=p_n)