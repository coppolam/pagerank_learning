#!/usr/bin/env python3
"""
Simulate the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""
import random, sys, pickle, matplotlib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
matplotlib.rc('text', usetex=True)
from deap import base, creator, tools
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
from . import evolution
from . import simulator
from tools import fileHandler as fh
import os
from . import desired_states_extractor

np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)

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
		
		# Inform the user of what's happening
		if verbose:
			print('{:=^40}'.format(' Start of evolution '))
			
		# Evaluate the initial population
		fitnesses = list(map(self.toolbox.evaluate, pop))
		for ind, fit in zip(pop, fitnesses):
			ind.fitness.values = fit

		# Start the evoution
		g = len(self.stats) # Number of generations
		gmax = len(self.stats) + generations # Max number to reach
		while g < gmax:
				
			# Determine offspring
			offspring = self.toolbox.select(pop, len(pop))
			offspring = list(map(self.toolbox.clone, offspring))
			for child1, child2 in zip(offspring[::2], offspring[1::2]):
				if random.random() < self.P_CROSSOVER: 
					self.toolbox.mate(child1, child2)
				del child1.fitness.values
				del child2.fitness.values

			# Mutate
			for mutant in offspring:
				if random.random() < self.P_MUTATION: 
					self.toolbox.mutate(mutant)
				del mutant.fitness.values
		
			# Evaluate the individuals with an invalid fitness
			invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
			fitnesses = map(self.toolbox.evaluate, invalid_ind)
			for ind, fit in zip(invalid_ind, fitnesses):
				ind.fitness.values = fit
			
			# Replace population
			pop[:] = offspring

			#################################################################
			# Load log files from evolution
			files = [f for f in os.listdir(self.temp_folder) \
												if f.endswith('.npz')]

			# Update transition model
			# The very first time, initialize the model, then just update it
			for j, f in enumerate(files):
				if g is 0 and j is 0:
					# This is the very first time
					self.sim.load(self.temp_folder + f, verbose=False)
				else:
					# From now on, we just update
					self.sim.load_update(self.temp_folder + f)

			self.sim.disp()

			# Now we optimize the policy
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

			## Model-based optimization
			p = list(self.sim.optimize(p, 
							settings=settings, model=model).flatten())

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
			self.stats.append(self.store_stats(pop, g)) 

			# Print to terminal
			if verbose: self.disp_stats(g)
			
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