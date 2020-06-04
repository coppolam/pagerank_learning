#!/usr/bin/env python3
"""
Simulate the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""

import pickle, sys, matplotlib, os, argparse
import matplotlib.pyplot as plt
import numpy as np
import aggregation as env
import evolution
matplotlib.rc('text', usetex=True)
import desired_states_extractor

###########################
#  Input argument parser  #
###########################
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('file', type=str, help="File to use")
parser.add_argument('controller', type=str, help="Controller to use")
parser.add_argument('agent', type=str, help="Agent to use")
parser.add_argument('-t', type=int, help="Max time of simulation. Default = 10000s", default=10000)
parser.add_argument('-n', type=int, help="Size of swarm. Default = 30", default=30)
parser.add_argument('-id', type=int, help="ID", default=1)
parser.add_argument('-observe', type=bool, help="", default=False)
args = parser.parse_args()

# Load environment
sim = env.aggregation()
folder = "data/learning_data_%s_%s/" % (args.controller,args.agent)

# run = False

def analyze_fitness(foldername):
	file = foldername + "1_learning_data_t%i_r%i.npz"
	# Load all and re-evaluate global and local fitnesses
	data = {"t":[], "f":[], "s":[]}
	sim.load(file=(file %(args.t,args.n)))
	# sim.sim.plot_log(file=(file %(args.t,c)))
	from tools import fitness_functions as ff
	t, f, s = sim.reevaluate(ff.number_of_clusters, ff.mean_number_of_neighbors)
	data["t"].append(t)
	data["f"].append(f)
	data["s"].append(s)
	return data

def compare_fitness(data):
	symbols = ["o",".","+","."]
	s = len(data["f"])
	corr = np.zeros((1,s))[0]
	for c in range(0,len(data["f"])):
		plt.plot(1/data["f"][c][:,0],data["f"][c][:,1],symbols[c],label=c)
		corr[c] = np.corrcoef(data["f"][c][:,0],data["f"][c][:,1])[0,1]
	plt.xlabel("Global fitness")
	plt.ylabel("Local fitness")
	plt.legend()
	plt.savefig(folder + "figures/fitness_comparison.pdf")
	plt.clf()
	return corr

# Save
def save_pkl(var,name):
	with open(name, "wb") as cp_file:
		pickle.dump(var, cp_file)

def load_pkl(name):
	with open(folder+"fitness_eval.pkl", "rb") as cp_file:
		data = pickle.load(cp_file)
	return data

def optimize(file,p0,des):
	sim.load(file)
	return sim.optimize(p0,des)

def benchmark(file,time_limit=100):
    # reference
	if args.controller == "aggregation":
		fitness = "aggregation_clusters"
		p_0 = np.ones((8,1))/2 # all = 1/2
	elif args.controller == "pfsm_exploration":
		fitness = "aggregation_clusters"
		p_0 = np.ones((16,8))/8 # all = 1/8
	elif args.controller == "pfsm_exploration_mod":
		fitness = "aggregation_clusters"
		p_0 = np.ones((16,8))/8
	elif args.controller == "forage":
		fitness = "food"
		p_0 = np.ones((16,1))/2
	else:
		ValueError("Uknown inputs!")

	des = desired_states_extractor.desired_states_extractor().run(file,verbose=True)
	p_n = optimize(file, p_0, des)
	if args.observe: sim.observe(p_0,args.controller,args.agent,robots=args.n)
	else:
		f_0 = sim.benchmark(p_0,args.controller,args.agent,fitness,robots=args.n,runs=100,time_limit=time_limit)
		f_n = sim.benchmark(p_n,args.controller,args.agent,fitness,robots=args.n,runs=100,time_limit=time_limit)
		data_validation = np.savez(folder + "benchmark_%s"%file,f_0=f_0,f_n=f_n,p_0=p_0,p_n=p_n)

	# e = evolution.evolution()
	# e.load(folder+"evolution")
	# p_s = e.get_best()
	# p_s = np.reshape(p_s,(16,8))
	# f_s = sim.benchmark(p_s,args.controller,args.agent,time_limit=time_limit)
	# data_validation = np.savez(folder + "benchmark.npz",f_0=f_0,f_n=f_n,f_s=f_s,p_0=p_0,p_n=p_n,p_s=p_s)
	
def plot_benchmark(file):
	data = np.load(folder + "benchmark_%s"%file)
	alpha = 0.5
	if "f_0" in data.files: plt.hist(data["f_0"].astype(float), alpha=alpha, label='$\pi_0$')
	if "f_n" in data.files: plt.hist(data["f_n"].astype(float), alpha=alpha, label='$\pi_n$')
	if "f_s" in data.files: plt.hist(data["f_s"].astype(float), alpha=alpha, label='$\pi*$')
	plt.legend()
	directory = os.path.dirname(folder + "figures/")
	if not os.path.exists(directory):
		os.makedirs(directory)
	plt.savefig(folder+"figures/benchmark_%s.pdf"%file)
	plt.clf()

def plot_evolution():
	e = evolution.evolution()
	e.load(folder+"evolution")
	e.plot_evolution(folder+"evolution_2.pdf")

benchmark(args.file,time_limit=1000)
# plot_benchmark(args.file)
# plot_evolution()
