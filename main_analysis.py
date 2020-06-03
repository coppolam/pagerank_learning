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
	if args.controller == "aggregation":
		fitness = "aggregation_clusters"
		p_0 = np.ones((8,1))/2 # all = 1/2
		# des = np.zeros([1,8])[0]
		# des[4:] = 1 # from nn
	elif args.controller == "pfsm_exploration":
		fitness = "aggregation_clusters"
		p_0 = np.ones((16,8))/8 # all = 1/8
		# des = np.ones([1,16])[0]
		# des[0] = 0 # from nn
	elif args.controller == "pfsm_exploration_mod":
		fitness = "aggregation_clusters"
		p_0 = np.ones((16,8))/8 # all = 1/8
		# des = np.ones([1,16])[0]
		# des[0] = 0 # from nn
		# des = [0.0, 0.9848900061808971, 0.976980500207466, 0.9891087896631704, 0.980456436324312, 0.9935526779444446, 0.9865328989055827, 0.991669223914925, 0.9754366280896926, 0.9634689362627317, 0.9923385484054741, 0.9995649232133225, 0.9128809930185515, 0.9990481518590899, 0.9847521337659593, 0.9949568797324759]
	elif args.controller == "forage":
		fitness = "food"
		p_0 = np.ones((16,1))/2
		# des = [0.9930061658199832, 0.9952269357749232, 0.9701479122355184, 0.9180163981328413, 0.9953555881858656, 0.9958821520830611, 0.9850341184365603, 0.9597566822246582, 1.0, 1.0, 0.9812966712145966, 0.9955001082675093, 0.0, 0.9988234867265866, 0.0, 0.0]
		# des = [0.9924909975428292, 0.995950167064943, 1.0, 1.0, 0.9917787816675288, 0.9977750734814962, 0.9539859077977527, 0.9987997668920423, 0.0, 0.9990176940934485, 0.9865668167518789, 0.0, 0.0, 0.0, 0.0, 0.0]
	else:
		ValueError("Uknown inputs!")

	des = desired_states_extractor.desired_states_extractor().run(file,verbose=True)
	p_n = optimize(file, p_0, des)

	# e = evolution.evolution()
	# e.load(folder+"evolution")
	# p_s = e.get_best()
	# p_s = np.reshape(p_s,(16,8))
	# f_0 = sim.benchmark(p_0,args.controller,args.agent,fitness,runs=100,time_limit=time_limit)
	# f_n = sim.benchmark(p_n,args.controller,args.agent,fitness,runs=100,time_limit=time_limit)
	sim.observe(p_n,args.controller,args.agent)
	# f_s = sim.benchmark(p_s,args.controller,args.agent,time_limit=time_limit)
	# data_validation = np.savez(folder + "benchmark.npz",f_0=f_0,f_n=f_n,f_s=f_s,p_0=p_0,p_n=p_n,p_s=p_s)
	# data_validation = np.savez(folder + "benchmark_%s"%file,f_0=f_0,f_n=f_n,p_0=p_0,p_n=p_n)

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

benchmark(args.file,time_limit=200)
plot_benchmark(file)
# plot_evolution()
