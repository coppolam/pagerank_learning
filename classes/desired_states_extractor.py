#!/usr/bin/env python3
'''
Optimize a behavior based on the PageRank function
@author: Mario Coppola, 2020
'''

import torch, os
import numpy as np
from . import network, evolution, simulator
from tools import matrixOperations as matop
from tools import fileHandler as fh
from . import simplenetwork
from deap import tools, creator

class desired_states_extractor:
	'''Trains a micro-macro link and extracted the desired observation set'''
	
	def __init__(self): 
		'''Initialize with an empty network'''
		self.network = None

	def train_model(self, x, y, layers=3, layer_size=30, lr=1e-5):
		'''
		Generate and/or train model using stochastic gradient descent.
		Train inputs x to match output y.
		'''
		
		# Create an empty network if it does not exist
		if self.network is None:
			print("Network model does not exist, generating the NN")
			self.network = network.net(n_inputs=x.shape[1],
										n_outputs=1,
										layers=layers,
										layer_size=layer_size,
										lr=lr)

		# Train over all data
		## TODO: Add variable batch sizes
		loss_history = []
		for i, element in enumerate(y):
			# Set up tensors
			in_tensor = torch.tensor([x[i]]).float()
			out_tensor = torch.tensor([[element]]).float()
			
			# Run training steps
			_,loss = self.network.run(in_tensor,out_tensor)

			# Store loss
			loss_history = np.append(loss_history,loss.item())

		return self.network, loss_history
	
	def evaluate_model(self,network,x,y):
		'''Evaluate the model against validation data'''

		# Set up a prediction list
		y_pred = []

		# Iterate over all test data
		for element in x:
			in_tensor = torch.tensor([element]).float()
			y_pred = np.append(y_pred, network.network(in_tensor).item())

		# Get the error vector
		error = y_pred - y

		# Get the Pearson correlation, [-1, 1]
		corr = np.corrcoef(y_pred, y)

		# If the values are exactly the same throughout and constant, 
		# then we have a nan problem, but this is actually good correlation
		# for our purposes.
		if np.isnan(corr[0][1]):
			corr[0][1] = 1.
		
		# Return tuple with outputs
		return error, corr[0][1], y_pred

	def load_model(self, modelsfile, modelnumber=-1):
		'''Load the latest model from the trained pkl file
		
		default modelnumber = -1, which is the last (assumed newest) 
								  model in the list

		It then returns the model
		'''
		# Load the file with all the models
		try:
			m = fh.load_pkl(modelsfile)
		except:
			print("Model not specified!")

		# Set the desired model
		# (default=-1, highest on the list, assumed newest)
		self.network = m[modelnumber][0]

		return self.network

	def extract_states(self, file, load_pkl=False, store_pkl=True):
		'''Extract the inputs needed to maximize output'''
		
		# If a pkl file does not exist, then we still need to do some dirty
		# work and load everything from the log files.
		# We will also store the pkl version to save time in future runs.
		if load_pkl is False or os.path.exists(file+".pkl") is False:
    		# Pre-process data
			sim = simulator.simulator() # Environment
			sim.load(file,verbose=False) # Load npz log file
			time, local_states, fitness = sim.extract() # Pre-process data
			s = matop.normalize_rows(local_states) # Normalize rows

			# Save a pkl file with the pre-processed data
			# so that we can be faster later
			# if we want to reuse the same logfile
			if store_pkl:
				fh.save_pkl([time,s,fitness],file+".pkl")

		# If the pkl file exists, we are in luck, we can just 
		# use the processed log files directly.
		else:
			data = fh.load_pkl(file+".pkl")
			time = data[0]
			s = data[1]
			fitness = data[2]
		
		# Set dimensions of state vector
		self.dim = s.shape[1]

		# Return tuple with data
		return time, s, fitness
		
	def _fitness(self,individual):
		'''Fitness function for the evolution'''

		# Normalize
		# individual = matop.normalize_rows(individual)

		# Set up tensor
		in_tensor = torch.tensor([individual]).float()

		# Return estimated fitness from network
		return self.network.network(in_tensor).item(),

	def get_des(self, dim=None, popsize=100, gens=500, debug=False):
		'''
		Runs an evolutionary optimization to extract 
		the states that maximize the fitness
		''' 
		
		# Initialize evolution
		e = evolution.evolution()
		
		# Set desired dimensions if specified
		if dim is None:
			d = self.dim
		else:
			d = dim
		
		# Set up boolean evolution
		e.setup(self._fitness, GENOME_LENGTH=d, 
				POPULATION_SIZE=popsize,vartype="boolean")

		# Evolve
		e.evolve(verbose=True, generations=gens)

		# Get desired observation set
		des = e.get_best()

		# Show a plot of the evolution, if plot=True
		if debug:
			e.plot_evolution()
		
		return des

	def train(self, file, load_pkl=True, store_pkl= True, 
				verbose=False, replay=1,
				layers=3, layer_size=100, lr=1e-6):
		'''
		Trains a model based on an npz simulation log file
		
		Use replay to re-iterate over the same data
		'''

		# Extract t = time, o = local observations, f = global fitness
		t, o, f = self.extract_states(file, 
							load_pkl=load_pkl,
							store_pkl=store_pkl)

		# Optimize to get the desired observation set
		if verbose:
			print("Training the NN model")
		
		# Train the feedforward model
		for i in range(replay):	
			model = self.train_model(o, f,
			layers=layers,
			layer_size=layer_size,
			lr=lr)
		
		return model

	def run(self, file, load=True, replay=1, verbose=False):
		'''
		Run the whole process with one function: 
		1) Train, 
		2) Get desired set Odes

		Use replay to re-iterate over the same data
		'''

		# Train the model
		self.train(file, replay=replay)

		# Get the desired states from the trained network
		if verbose:
			print("Optimizing for desired observations")
		
		des = self.get_des(plot=verbose, popsize=500)
		
		if verbose:
			print("Desired observations: " + str(des))
		
		return des