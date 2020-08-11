#!/usr/bin/env python3
"""
Perform evolution for a given behavior
	Run as: python3 main_standard_evolution.py CONTROLLER [-opt]
	Example: python3 main_standard_evolution.py aggregation [-opt]
@author: Mario Coppola, 2020
"""

import random, sys, pickle, argparse, os
import numpy as np
from tools import fileHandler as fh
from classes import evolution
from simulators import swarmulator, parameters
from tools import matrixOperations as matop

# Argument parser
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('controller', type=str, help="Controller to use")
parser.add_argument('-generations', type=int, help="Max generations", default=50)
parser.add_argument('-pop', type=int, help="Population size", default=100)
parser.add_argument('-t', type=int, help="Time", default=200)
parser.add_argument('-nmin', type=int, help="Time", default=10)
parser.add_argument('-nmax', type=int, help="Time", default=20)
parser.add_argument('-reruns', type=int, help="Batch size", default=5)
parser.add_argument('-plot', type=str, help="", default=None)
parser.add_argument('-environment', type=str, help="environment", default="square20")
parser.add_argument('-id', type=int, help="Evo ID", default=np.random.randint(0,10000))
parser.add_argument('-resume', action='store_true', help="(bool) Animate flag to true")
args = parser.parse_args()

# Load parameters
fitness, controller, agent, pr_states, pr_actions = parameters.get(args.controller)

# Set up directory
folder = "data/%s/" % (controller)
directory = os.path.dirname(folder)
if not os.path.exists(directory): os.makedirs(directory)

# Fitness function
def fitnessfunction(individual):
	'''Fitness function that evaluates the average performance of the individuals'''
	
	# Set the policy file that swarmulator reads
	policy_file = "../swarmulator/conf/state_action_matrices/policy_evolved_temp.txt"
	if pr_actions > 1: # Reshape and normalize along rows
		individual = np.reshape(individual,(pr_states,pr_actions))
		individual = matop.normalize_rows(individual)
	fh.save_to_txt(individual, policy_file)
	sim.runtime_setting("policy", policy_file) # Use random policy

	# Run swarmulator in batches
	f = []
	for i in range(args.reruns):
		if args.nmin < args.nmax: robots = np.random.randint(args.nmin,args.nmax) # Random number of robots within bounds
		else: robots = args.nmin
		f = np.append(f,sim.run(robots)) # Simulate
	# f = sim.batch_run((args.nmin,args.nmax),args.reruns) # Run with 10-20 agents
	
	print(f) # Just to keep track
	return f.mean(), # Fitness = average (note trailing comma to cast to tuple!)

# Load evolution API
e = evolution.evolution()
e.setup(fitnessfunction, GENOME_LENGTH=pr_states*pr_actions, POPULATION_SIZE=args.pop)

# Plot file from file args.plot
if args.plot is not None:
	e.load(args.plot)
	e.plot_evolution()
	print(e.get_best())
	exit()
	
# Swarmulator API
sim = swarmulator.swarmulator(verbose=False)
sim.make(controller=controller, agent=agent, animation=False, clean=True, logger=False, verbose=False)
sim.runtime_setting("time_limit", str(args.t))
sim.runtime_setting("simulation_realtimefactor", str("300"))
sim.runtime_setting("environment", args.environment)
filename = folder + "evolution_%s_t%i_%i" % (controller, args.t, args.id)

# Simulation parameters
sim.runtime_setting("fitness", fitness)

# Resume evolution from file args.resume
if args.resume is True:
	e.load(args.resume)
	p = e.evolve(verbose=True, generations=args.generations, checkpoint=filename, population=e.pop)

# Just run normally
else: p = e.evolve(verbose=True, generations=args.generations, checkpoint=filename)

# Save
e.save(filename)
