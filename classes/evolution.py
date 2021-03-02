#!/usr/bin/env python3
"""
Wrapper around the DEAP package to run an evolutionary process with just a few commands

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

np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)

class evolution:
	'''
	Wrapper around the DEAP package to run an 
	evolutionary process with just a few commands
	'''

	def __init__(self):
		'''Itialize the DEAP wrapper'''
		pass
	
	def setup(self, fitness_function_handle, 
		constraint=None,
		GENOME_LENGTH=20,
		POPULATION_SIZE=100, 
		P_CROSSOVER=0.5, 
		P_MUTATION=0.2,
		vartype="float"):
		'''Set up the standard parameters'''
		
		# Set the main variables
		self.GENOME_LENGTH = GENOME_LENGTH
		self.POPULATION_SIZE = POPULATION_SIZE
		self.P_CROSSOVER = P_CROSSOVER
		self.P_MUTATION = P_MUTATION

		# Set the lower level parameters
		self.toolbox = base.Toolbox()
		
		# Boolean or float evolution
		if vartype=="boolean":
			self.toolbox.register("attr_bool",
							random.randint, 0, 1)   			
			self.toolbox.register("individual", 
							tools.initRepeat,
							creator.Individual, 
							self.toolbox.attr_bool,
							GENOME_LENGTH)
			self.toolbox.register("mutate", 
							tools.mutUniformInt, 
							low=0,
							up=1,
							indpb=0.05)
		elif vartype=="float":
			self.toolbox.register("attr_float", 
							random.random)
			self.toolbox.register("individual", 
							tools.initRepeat, 
							creator.Individual, 
							self.toolbox.attr_float, 
							self.GENOME_LENGTH)
			
			self.toolbox.register("mutate", 
							tools.mutPolynomialBounded, 
							eta=0.1, 
							low=0.0, 
							up=1.0,
							indpb=0.1)

		self.toolbox.register("population", 
			tools.initRepeat, 
			list, 
			self.toolbox.individual)

		self.toolbox.register("evaluate", 
			fitness_function_handle)

		self.toolbox.register("mate", 
			tools.cxUniform, indpb=0.1)
		
		self.toolbox.register("select", 
			tools.selTournament, 
			tournsize=3)

		if constraint is not None:
				self.toolbox.decorate("evaluate", 
					tools.DeltaPenalty(constraint,0,self.distance))
	
		self.stats = [] # Initialize stats vector
	
	def store_stats(self, population, iteration=0):
		'''Store the current stats and return a dict'''
		# Gather the fitnesses in a population
		fitnesses = [individual.fitness.values[0] for individual in population]

		# Store the main values
		return {
			'g': iteration,
			'mu': np.mean(fitnesses),
			'std': np.std(fitnesses),
			'max': np.max(fitnesses),
			'min': np.min(fitnesses)
		}

	def disp_stats(self,iteration=0):
		'''Print the current stats to the terminal'''

		# String to print
		p = "\r >> gen = %i, mu = %.2f, std = %.2f, max = %.2f, min = %.2f" % (
			self.stats[iteration]['g'],
			self.stats[iteration]['mu'],
			self.stats[iteration]['std'],
			self.stats[iteration]['max'],
			self.stats[iteration]['min'])

		# Sys write + flush
		sys.stdout.write(p)
		sys.stdout.flush()

	def plot_evolution(self,figurename=None):
		'''Plot the evolution outcome'''
		
		# Set up latex
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		plt.figure(figsize=(6,3))
		plt.gcf().subplots_adjust(bottom=0.15)
		
		# Plot
		plt.plot(range(1, len(self.stats)+1), 
			[ s['mu'] for s in self.stats ],
			label="Mean")
		plt.fill_between(range(1, len(self.stats)+1),
					[ s['min'] for s in self.stats ],
					[ s['max'] for s in self.stats ],
					color='green',
					alpha=0.2,
					label="Min Max")
		plt.fill_between(range(1, len(self.stats)+1),
					[ s['mu']-s['std'] for s in self.stats ],
					[ s['mu']+s['std'] for s in self.stats ],
					color='gray',
					alpha=0.5,
					label="Std")
		plt.xlabel('Iterations')
		plt.ylabel('Fitness')
		plt.legend()
		plt.xlim(0,len(self.stats))

		# Save if a figurename was given, else just show it
		if figurename is not None:
			plt.savefig(figurename)
			plt.close()
		else:
			plt.show()
			plt.close()

	def evolve(self, generations=100, 
				verbose=False, population=None, checkpoint=None):
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

	def save(self,filename,pop=None,gen=None,stats=None):
		'''Save the current status in a pkl file'''
		
		# Population
		p = self.pop if pop is None else pop
		
		# Generation
		g = self.g if gen is None else gen
		
		# Statistics
		s = self.stats if stats is None else stats
		
		# Store in a dict file and save as pkl
		cp = dict(population=p, generation=g, stats=s)
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

		# Return the population
		return self.pop

	def get_best(self,pop=None):
		'''Returns the fittest element of a population'''
		# Get population
		p = self.pop if pop is None else pop

		# Return the best
		return tools.selBest(p,1)[0]
