#!/usr/bin/env python3
"""
Optimize a behavior based on the PageRank function
@author: Mario Coppola, 2020
"""
import torch, os
import numpy as np
from tqdm import tqdm
from classes import simplenetwork, evolution, simulator
from tools import matrixOperations as matop
from tools import fileHandler as fh

class desired_states_extractor:
	def __init__(self): 
		self.network = None

	def train_model(self,x,y):
		''' Generate and/or train model using stochastic gradient descent'''
		if self.network is None:
			print("Model does not exist, generating the NN")
			self.network = simplenetwork.simplenetwork(x.shape[1])
		loss_history = []
		i = 0
		for element in y:
			in_tensor = torch.tensor([x[i]]).float()
			out_tensor = torch.tensor([[element]]).float()
			_,loss = self.network.run(in_tensor,out_tensor)
			loss_history = np.append(loss_history,loss.item())
			i += 1
		return self.network, loss_history
	
	def evaluate_model(self,network,x,y):
		'''Evaluate the model against validation data'''
		y_pred = []
		for element in x:
			in_tensor = torch.tensor([element]).float()
			y_pred = np.append(y_pred,network.network(in_tensor).item())
		error = y_pred - y
		corr = np.corrcoef(y_pred,y)
		return error, corr, y_pred

	def load_model(self,modelsfile):
		'''Load the latest model from the trained pkl file'''
		m = fh.load_pkl(modelsfile)
		self.network = m[-1][0] # last model

	def extract_states(self,file,pkl=False):
		'''Extract the inputs needed to maximize output'''
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
		# individual = matop.normalize_rows(individual,axis=0)
		'''Fitness function'''
		in_tensor = torch.tensor([individual]).float()
		f = self.network.network(in_tensor).item()
		return f, 

	# def _constraint(self,individual):
	# 	'''Feasibility function for the individual. Returns True if feasible False otherwise.'''
	# 	if np.sum(individual) < 1: return True
	# 	return False

	def get_des(self,dim=None):
		'''Runs an evolutionary optimization to extract the states that maximize the fitness''' 
		e = evolution.evolution()
		if dim is None: d = self.dim
		else: d = dim
		e.setup(self._fitness, GENOME_LENGTH=d, POPULATION_SIZE=1000)
		e.evolve(verbose=True, generations=100)
		des = e.get_best()
		# e.plot_evolution()#"%s_evo_des.pdf"%file)
		return des

	def run(self,file,load=True,verbose=False, replay=1):
		'''Runs everything'''
		t, s, f = self.extract_states(file, pkl=load)
		if verbose: print("Training the NN model")
		for i in range(replay): self.train_model(s, f)
		if verbose: print("Optimizing for desired states")
		des = self.get_des()
		if verbose: print("Desired states: " + str(des))
		return des

	def train(self, file, load=True, verbose=False, replay=1):
		'''Trains a model based on an npz simulation log file'''
		t, s, f = self.extract_states(file, pkl=load)
		if verbose: print("Training the NN model")
		for i in range(replay): model = self.train_model(s, f)
		return model