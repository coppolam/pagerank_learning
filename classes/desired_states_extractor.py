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
import scipy
from scipy.special import softmax
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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
		'''Fitness function'''
		in_tensor = torch.tensor([individual]).float()
		f = self.network.network(in_tensor).item()
		return f, 

	def get_des(self):
		e = evolution.evolution()
		e.setup(self._fitness, GENOME_LENGTH=self.dim, POPULATION_SIZE=1000)
		e.evolve(verbose=False, generations=100)
		des = e.get_best()
		e.plot_evolution()#"%s_evo_des.pdf"%file)
		return des

	def run(self,file,load=True,verbose=False, replay=1):
		t, s, f = self.extract_states(file, pkl=load)
		if verbose: print("Training the NN model")
		for i in range(replay): self.train_model(s, f)
		if verbose: print("Optimizing for desired states")
		des = self.get_des()
		if verbose: print("Desired states: " + str(des))
		return des

	def train(self, file, load=True, verbose=False, replay=1):
		t, s, f = self.extract_states(file, pkl=load)
		if verbose: print("Training the NN model")
		for i in range(replay): model = self.train_model(s, f)
		return model