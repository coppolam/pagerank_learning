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
parser.add_argument('-generations', type=int, help="Max generations after which the evolution quits, default = 50", default=50)
parser.add_argument('-pop', type=int, help="Population size used in the evolution, default = 100", default=100)
parser.add_argument('-t', type=int, help="Time for which each simulation is executed, deafult = 200s", default=200)
parser.add_argument('-nmin', type=int, help="Minimum number of robots simulated, default = 10", default=10)
parser.add_argument('-nmax', type=int, help="Maximum number of robots simulated, default = 20", default=20)
parser.add_argument('-reruns', type=int, help="Number of policy re-evaluations , default = 5", default=5)
parser.add_argument('-plot', type=str, help="Specify the relative path to a pkl evolution file to plot the evolution.", default=None)
parser.add_argument('-environment', type=str, help="Environment used in the simulations, default is a square room of size 20 by 20.", default="square20")
parser.add_argument('-id', type=int, help="ID of evolutionary run, default = random integer", default=np.random.randint(0,10000))
parser.add_argument('-resume', type=str, help="If specified, it will resume the evolution from a previous pkl checkpoint file", default=None)
parser.add_argument('-evaluate', type=str, help="If specified, it will evaluate the best result in an evolution pkl file", default=None)
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
	else:
		individual = np.reshape(individual,(1,pr_states))	
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
sim.make(controller=controller, agent=agent, animation=True, clean=True, logger=False, verbose=False)
sim.runtime_setting("time_limit", str(args.t))
sim.runtime_setting("simulation_realtimefactor", str("300"))
sim.runtime_setting("environment", args.environment)
filename = folder + "evolution_%s_t%i_%i" % (controller, args.t, args.id)

# Simulation parameters
sim.runtime_setting("fitness", fitness)

if args.evaluate is not None:
	e.load(args.evaluate)
	individual = e.get_best()
	# Set the policy file that swarmulator reads
	policy_file = "../swarmulator/conf/state_action_matrices/policy_evolved_temp.txt"
	if pr_actions > 1: # Reshape and normalize along rows
		individual = np.reshape(individual,(pr_states,pr_actions))
		individual = matop.normalize_rows(individual)
	else:
		individual = np.reshape(individual,(1,pr_states))	
	fh.save_to_txt(individual, policy_file)
	sim.runtime_setting("policy", policy_file) # Use random policy
	# Uncomment to just observe what happens
	# sim.runtime_setting("time_limit", "0") # Use random policy
	# np. set_printoptions(suppress=True)
	# print(individual)
	# sim.run(args.nmax)
	f = []
	for i in range(args.reruns):
		if args.nmin < args.nmax: robots = np.random.randint(args.nmin,args.nmax) # Random number of robots within bounds
		else: robots = args.nmin
		f = np.append(f,sim.run(args.nmax)) # Simulate
	
	fh.save_pkl(f,"data/%s/benchmark_evolution_%s_t%i_r%i_runs%i.pkl"%(controller,controller,args.t,args.nmax,args.reruns))
	
	exit()
	
# Resume evolution from file args.resume
if args.resume is not None:
	e.load(args.resume)
	p = e.evolve(verbose=True, generations=args.generations, checkpoint=filename, population=e.pop)

# Just run normally
else: p = e.evolve(verbose=True, generations=args.generations, checkpoint=filename)

# Save
e.save(filename)
