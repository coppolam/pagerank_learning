#!/usr/bin/env python3
"""
Optimize a behavior based on the PageRank function
@author: Mario Coppola, 2020
"""
import torch, os
import numpy as np
from tqdm import tqdm
from tools import matrixOperations as matop
from classes import simplenetwork, evolution, simulator
from tools import fileHandler as fh

class desired_states_extractor:
	def __init__(self):
		pass

	def make_model(self,x,y):
		self.network = simplenetwork.simplenetwork(x.shape[1])
		i = 0
		loss_history = []
		for element in tqdm(y):
			in_tensor = torch.tensor([x[i]]).float()
			out_tensor = torch.tensor([[element]]).float()
			_,loss = self.network.run(in_tensor,out_tensor)
			loss_history = np.append(loss_history,loss.item())
			i += 1
		return self.network, loss_history
	
	def evaluate_model(self,network,x,y):
		y_pred = []
		for element in tqdm(x):
			in_tensor = torch.tensor([element]).float()
			y_pred = np.append(y_pred,network.network(in_tensor).item())
		error = y_pred - y
		return error

	def extract_states(self,file,pkl=False):
		if pkl is False or os.path.exists(file+".pkl") is False:
			sim = simulator.simulator()
			sim.load(file)
			time, local_states, fitness = sim.extract()
			s = matop.normalize_rows(local_states)
			fh.save_pkl([time,s,fitness],file+".pkl")
		else:
			data = fh.load_pkl(file+".pkl")
			time = data[0]
			s = data[1]
			fitness = data[2]
		self.dim = s.shape[1]
		return time, s, fitness
		
	def _fitness(self,individual):
		in_tensor = torch.tensor([individual]).float()
		f = self.network.network(in_tensor).item()
		return f, 

	def get_des(self):
		e = evolution.evolution()
		e.setup(self._fitness, GENOME_LENGTH=self.dim, POPULATION_SIZE=1000)
		p = e.evolve(verbose=False, generations=100)
		return e

	def run(self,file,load=True,verbose=False):
		t, s, f = self.extract_states(file,pkl=load)

		if verbose: print("Making the NN model")
		model = self.make_model(s, f)
		
		if verbose: print("Optimizing for desired states")
		s = self.get_des()
		des = s.get_best()
		
		if verbose: print("Desired states: " + str(des))
		
		return des
