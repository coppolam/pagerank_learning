#!/usr/bin/env python3
"""
Simulate the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""

import aggregation as env
import pickle, sys
import matplotlib.pyplot as plt
import numpy as np
import evolution
import matplotlib
matplotlib.rc('text', usetex=True)

# Load environment
sim = env.aggregation()
run = False
folder = sys.argv[1]
r = [10,20,30]
tmax = 10000

def analyze(foldername):
	file = foldername + "1_learning_data_t%i_r%i.npz"
	from tools import fitness_functions as ff

	# Load all and re-evaluate global and local fitnesses
	counter = 0
	data = {"t":[], "f":[], "s":[]}
	for c in r:
		sim.load(file=(file %(tmax,c)))
		# sim.sim.plot_log(file=(file %(tmax,c)))
		t, f, s = sim.reevaluate(ff.number_of_clusters, ff.mean_number_of_neighbors)
		data["t"].append(t)
		data["f"].append(f)
		data["s"].append(s)
		counter += 1

	return data

def compare(data):
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

def optimize(foldername,tmax,r,des):
	file = foldername + "1_learning_data_t%i_r%i.npz"
	sim.load(file=(file %(tmax,r)))
	sim.disp()
	
	# des = np.zeros([1,16])[0]
	# des[14] = 1
	des = np.zeros([1,8])[0]
	des[4] = 1

	policy = sim.optimize(des)
	return policy

def benchmark(time_limit=100):
	p_n = optimize(folder,tmax,20)
	p_0 = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
	# p_0 = np.ones((16,8))/8
	e = evolution.evolution()
	e.load(folder+"evolution")
	p_s = e.get_best()
	# p_s = np.reshape(p_s,(16,8))

	f_0 = sim.benchmark(p_0,time_limit=time_limit)
	f_n = sim.benchmark(p_n,time_limit=time_limit)
	f_s = sim.benchmark(p_s,time_limit=time_limit)
	data_validation = np.savez(folder + "benchmark.npz",f_0=f_0,f_n=f_n,f_s=f_s,p_0=p_0,p_n=p_n,p_s=p_s)

def plot_benchmark():
	data = np.load(folder + "benchmark.npz")
	f_0 = data["f_0"].astype(float)
	f_n = data["f_n"].astype(float)
	f_s = data["f_s"].astype(float)
	alpha = 0.5
	plt.hist(f_0, alpha=alpha, label='$\pi_0$')
	plt.hist(f_n, alpha=alpha, label='$\pi_n$')
	plt.hist(f_s, alpha=alpha, label='$\pi*$')
	plt.legend()
	plt.savefig(folder+"figures/benchmark.pdf")
	plt.clf()

def plot_evolution():
	e = evolution.evolution()
	e.load(folder+"evolution")
	e.plot_evolution(folder+"evolution_2.pdf")


# if run:
# 	fdata = analyze(folder)
# 	save_pkl(fdata,folder+"fitness_eval.pkl")

# benchmark()
# compare(load_pkl(folder+"fitness_eval.pkl"))
# plot_benchmark()
plot_evolution()
