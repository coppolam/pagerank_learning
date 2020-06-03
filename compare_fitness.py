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
import torch
from tools import fitness_functions as ff
from tools import matrixOperations as matop

###########################
#  Input argument parser  #
###########################
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('file', type=str, help="File to analyze", default=None)
parser.add_argument('-load', type=bool, help="Load from file or re-run", default=False)
args = parser.parse_args()

sim = env.aggregation()

class simplenetwork:
	def __init__(self,D_in):
		self.network = self.initialize(D_in, 1000, 1)
		self.loss_fn = torch.nn.MSELoss(reduction='sum')
		# self.optimizer = torch.optim.SGD(self.network.parameters(), lr=1e-4)
		self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4)

	def initialize(self, D_in, H, D_out):
		model = torch.nn.Sequential(
			torch.nn.Linear(D_in, H),
			torch.nn.ReLU(),
			# torch.nn.Linear(H, 5,
			# torch.nn.ReLU(),
			torch.nn.Linear(H, D_out))
		return model

	def run(self,x,y):
		y_pred = self.network(x)
		loss = self.loss_fn(y_pred, y)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return y_pred, loss

	def get(self):
		return self.network

def evaluate_fitness_from_log(file,*fitness_functions):
	# Load all and re-evaluate global and local fitnesses
	sim.load(file=file)
	data = {"time":[], "fitness":[], "local_states":[], "fitness_official":[]}
	time, fitness, local_states, fitness_official = sim.reevaluate(*fitness_functions)
	data["time"].append(time)
	data["fitness"].append(fitness)
	data["local_states"].append(local_states)
	data["fitness_official"].append(fitness_official)
	return data

def compare_fitness(data):
	symbols = ["o",".","+","."]
	s = len(data["fitness"])
	corr = np.zeros((1,s))[0]
	for c in range(0,len(data["fitness"])):
		plt.plot(1/data["fitness"][c][:,0],data["fitness"][c][:,1],symbols[c],label=c)
		corr[c] = np.corrcoef(data["fitness"][c][:,0],data["fitness"][c][:,1])[0,1]
	plt.xlabel("Global fitness")
	plt.ylabel("Local fitness")
	plt.legend()
	folder = os.path.dirname(args.file)

	## Filename
	folder = os.path.dirname(args.file)
	filename = os.path.basename(args.file)
	filename_raw = os.path.splitext(filename)[0]

	plt.savefig(folder + "/figures/" + filename_raw + "_fitness_comparison.pdf")
	plt.clf()
	return corr

def get_model(data):
	fg = data["fitness_official"][0] # to get the official fitness
	# print(f)
	s = matop.normalize_rows(data["local_states"][0])
	# fg = f[:,0]
	network = simplenetwork(s.shape[1])
	i = 0
	out = []
	for v in fg:
		in_tensor = torch.tensor([s[i]]).float()
		out_tensor = torch.tensor(v).float()
		action, loss = network.run(in_tensor,out_tensor)
		out = np.append(out,loss.item())
		i += 1
	plt.plot(range(0,i),out)
	plt.show()
	return network

def save_pkl(var,name): 
	with open(name, "wb") as cp_file: pickle.dump(var, cp_file)

def load_pkl(file):
	with open(file, "rb") as cp_file: data = pickle.load(cp_file)
	return data

## Filename
folder = os.path.dirname(args.file)
filename = os.path.basename(args.file)
filename_raw = os.path.splitext(filename)[0]
file = folder + "/" + filename_raw + "_fitness_eval.pkl"

if args.load is False:
	data = evaluate_fitness_from_log(args.file)#,ff.number_of_clusters)
	save_pkl(data,file)

data = load_pkl(file)
model = get_model(data)

# ## check against another sim
# network = model.get()
# filename_raw_check = filename_raw.replace(filename_raw[len(filename_raw)-1], '7')
# data_check_file = folder + "/" + filename_raw_check + ".npz"
# data_check = evaluate_fitness_from_log(data_check_file,ff.number_of_clusters)
# fg = data_check["fitness_official"][0]
# s = matop.normalize_rows(data_check["local_states"][0])

# # fg = f[:,0]
# i = 0
# out = []
# for v in fg:
# 	in_tensor = torch.tensor([s[i]]).float()
# 	out = np.append(out,network(in_tensor).item())
# 	i += 1
# plt.plot(range(0,i),out-fg)
# plt.show()

## now maximize fg
network = model.get()
def fitness(individual):
	in_tensor = torch.tensor([individual]).float()
	f = network(in_tensor).item()
	return f, 

e = evolution.evolution()
s = matop.normalize_rows(data["local_states"][0])
e.setup(fitness, GENOME_LENGTH=s.shape[1], POPULATION_SIZE=1000)
p = e.evolve(verbose=False, generations=100)
print(e.get_best())
e.plot_evolution()
