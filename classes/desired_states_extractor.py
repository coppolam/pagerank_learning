#!/usr/bin/env python3
"""
Optimize a behavior based on the PageRank function
@author: Mario Coppola, 2020
"""
import torch
import numpy as np
from tqdm import tqdm
from tools import matrixOperations as matop
from classes import simplenetwork, evolution, simulator

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
		for element in x:
			in_tensor = torch.tensor([element]).float()
			y_pred = np.append(y_pred,network.network(in_tensor).item())
		error = y_pred - y
		return error

	def extract_states(self,file):
		sim = simulator.simulator()
		sim.load(file)
		local_states, fitness = sim.extract()
		self.dim = local_states.shape[1]
		return matop.normalize_rows(local_states), fitness
		
	def _fitness(self,individual):
		in_tensor = torch.tensor([individual]).float()
		f = self.network.network(in_tensor).item()
		return f, 

	def get_des(self):
		e = evolution.evolution()
		e.setup(self._fitness, GENOME_LENGTH=self.dim, POPULATION_SIZE=1000)
		p = e.evolve(verbose=False, generations=100)
		return e.get_best()

	def run(self,file,verbose=False):
		if verbose: print("Extracting data from log")
		s, f = self.extract_states(file)
		
		if verbose: print("Making the NN model")
		model = self.make_model(s, f)
		
		if verbose: print("Optimizing for desired states")
		des = self.get_des()
		
		if verbose: print("Desired states: " + str(des))
		
		return des
