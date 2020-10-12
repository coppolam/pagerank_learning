#!/usr/bin/env python3
"""
Simulate the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""
import random, sys, pickle, os
import numpy as np
from tqdm import tqdm
import copy

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('text', usetex=True)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)

from deap import base, creator, tools

from . import evolution
from . import simulator
from . import desired_states_extractor
from tools import fileHandler as fh

class mbe(evolution.evolution):
	'''
	Wrapper around the evolution package to run a model based variant
	'''

	def __init__(self,model_temp_folder):
		'''Initialize'''

		# Load up the evolution class. 
		# We only wish to replace the evolve function
		super().__init__()

		# Set up the simulator API which includes the model-based optimization
		self.sim = simulator.simulator()

		# Desired states neural network model
		self.dse = desired_states_extractor.desired_states_extractor()
		
		# Set up a temp folder used to store the logs during the evolution
		self.temp_folder = model_temp_folder
		self.clear_model_data()
	
	def store_stats(self, population, iteration=0,
						transition_model=None,nn=None):
		'''
		Store the current stats and return a dict.
		In this subclass, also store the transition model and the NN model
		'''
		# Gather the fitnesses in a population
		fitnesses = [individual.fitness.values[0] for individual in population]

		# Store the main values
		return {
			'g': iteration,
			'mu': np.mean(fitnesses),
			'std': np.std(fitnesses),
			'max': np.max(fitnesses),
			'min': np.min(fitnesses),
			'transition_model': transition_model,
			'nn_model':nn
		}

	def _mate(self,pop):
		'''Mate the population and generate offspring'''
		offspring = self.toolbox.select(pop, len(pop))
		offspring = list(map(self.toolbox.clone, offspring))
		for child1, child2 in zip(offspring[::2], offspring[1::2]):
			if random.random() < self.P_CROSSOVER: 
				self.toolbox.mate(child1, child2)
			del child1.fitness.values
			del child2.fitness.values
		return offspring

	def _mutate(self,offspring):
		'''Mutate the offspring'''
		for mutant in offspring:
			if random.random() < self.P_MUTATION: 
				self.toolbox.mutate(mutant)
			del mutant.fitness.values
		return offspring
	
	def _evaluate(self,offspring):
		'''Evaluate the offspring with unknown fitness'''
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = map(self.toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit
		return offspring

	def evolve(self, generations=100, 
				verbose=False, 
				population=None, 
				checkpoint=None,
				settings=None,
				pretrained=False):
		'''
		Run the evolution. Use checkpoint="filename.pkl" to 
		save the status to a file after each generation, 
		just in case.
		'''

		# Set up a random seed
		random.seed() 

		# Initialize the population (if not given)
		pop = population if population is not None \
			else self.toolbox.population(n=self.POPULATION_SIZE)
		
		g = len(self.stats) # Number of generations
		gmax = len(self.stats) + generations # Max number to reach

		# If it's the first generation, evaluate the initial population
		if g is 0:
			# Evaluate the initial population
			fitnesses = list(map(self.toolbox.evaluate, pop))
			for ind, fit in zip(pop, fitnesses):
				ind.fitness.values = fit

		# Inform the user of what's happening
		if verbose:
			print('{:=^40}'.format(' Start of evolution '))
			
		# Start the evoution
		while g < gmax:
    			
    		# Create the next generation: 1) mate, 2) mutate, 3) evaluate
			offspring = self._mate(pop)
			offspring = self._mutate(offspring)
			offspring = self._evaluate(offspring)

			# Replace population
			pop[:] = offspring

			#################################################################
			# Model-based section.
			# We use the data from the population to extract 2 models:
			# 1) A model of the transitions
			# 2) A function relating global fitness to local states
			
			# Load log files from evolution
			files = [f for f in os.listdir(self.temp_folder) \
												if f.endswith('.npz')]

			# 1) Transition model
			# The very first time, initialize the model, then just update it
			for j, f in enumerate(files):
				if g is 0 and j is 0:
					# This is the very first time
					self.sim.load(self.temp_folder + f, verbose=False)
				else:
					# From now on, we just update
					self.sim.load_update(self.temp_folder + f, verbose=False)

			# 2) Neural network function
			## Initial policy to optimize from a random point
			p = np.random.rand(settings["pr_states"],settings["pr_actions"])

			## Train the feedforward model with new data
			if pretrained is False:
				for f in files:
					self.dse.train("data/evo_temp/"+f,
									load_pkl=False, store_pkl=False)
				model = self.dse.network
			else:
				model = self.dse.load_model("data/%s/models.pkl"%
										settings["controller"],modelnumber=499)

			## Use the models to do model-based optimization
			p = list(self.sim.optimize(p, 
							settings=settings,
							model=model, 
							debug=False).flatten())

			## Evaluate model-based solution
			### Create an individual
			ind = self.toolbox.individual()

			### Replace individual with the model optimized solution
			ind[:] = p

			### Evaluate individual
			ind.fitness.values = self.toolbox.evaluate(ind)

			## Replace optimized policy in population
			pop[-1] = ind

			## Clear temp folder
			self.clear_model_data()
			#################################################################

			# Store stats
			self.stats.append(self.store_stats(
				pop, 
				g, 
				copy.deepcopy(self.sim),
				copy.deepcopy(model))) 

			# Print to terminal
			if verbose:
				self.disp_stats(g)
			
			# Store progress
			if checkpoint is not None:
				self.save(checkpoint, pop=pop, gen=g, stats=self.stats)

			# Move to next generation
			g += 1

		# Store oucomes
		self.pop = pop
		self.best_ind = self.get_best()
		self.g = g

		# Save to checkpoint file
		if checkpoint is not None:
			self.save(checkpoint)

		# Display outcome
		if verbose: 
			print('\n{:=^40}'.format(' End of evolution '))
			print("Best individual is %s, %s" % (self.best_ind, self.best_ind.fitness.values))

		return pop

	def clear_model_data(self):
		fh.clear_folder(self.temp_folder)

	def save(self,filename,pop=None,gen=None,stats=None,sim=None,dse=None):
		'''Save the current status in a pkl file'''
		
		# Population
		p = self.pop if pop is None else pop
		
		# Generation
		g = self.g if gen is None else gen
		
		# Statistics
		s = self.stats if stats is None else stats
		
		# Transition model
		m = self.sim if sim is None else sim

		# NN model
		d = self.dse if dse is None else dse
		
		# Store in a dict file and save as pkl
		cp = dict(population=p, generation=g, stats=s, sim=m, dse=d)
		with open(filename+".pkl", "wb") as cp_file:
			pickle.dump(cp, cp_file)

	def load(self,filename):
		'''Load the status from a pkl file'''
		# Load a pkl file with the same structure as in the save() function
		with open(filename, "rb") as cp_file:
			cp = pickle.load(cp_file)
		
		# Unpack
		self.stats = cp["stats"]
		self.g = cp["generation"]
		self.pop = cp["population"]
		self.sim = cp["sim"]
		self.dse = cp["dse"]

		print("Loaded")
		self.sim.disp()
		print(self.dse.network)
		# Return the population
		return self.pop