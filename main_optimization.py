#!/usr/bin/env python3
"""
Load the training data + model and optimize the behavior accordingly.
Then evaluate it and save the results.

@author: Mario Coppola, 2020
"""

import pickle, os, argparse
import numpy as np
import parameters

from tools import fileHandler as fh
from tools import matrixOperations as matop
from classes.desired_states_extractor import desired_states_extractor as dse
from classes.simulator import simulator as sim
from classes.evolution import evolution as evo

np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)

if __name__ == "__main__":
	####################################################################
	# Initialize

	# Input argument parser
	parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
	parser.add_argument('controller', type=str, 
		help="(str) Controller to use during evaluation")
	parser.add_argument('folder_training', type=str, 
		help="(str) Controller to use during evaluation")
	parser.add_argument('-t', type=int, default=200,
		help="(int) Simulation time during benchmark, default = 200s")
	parser.add_argument('-n', type=int, default=30,
		help="(int) Size of swarm, default = 30")
	parser.add_argument('-runs', type=int, default=100, 
		help="(int) Evaluation runs, default = 100")
	parser.add_argument('-id', type=int, default=np.random.randint(1000),
		help="(int) ID of run, default = random")
	parser.add_argument('-iterations', type=int, default=0,
		help="(int) Number of iterations")
	parser.add_argument('-environment', type=str, default="square20",
		help="(str) Number of iterations", )
	parser.add_argument('-animate', action='store_true', 
		help="(bool) Animate flag to true")
	parser.add_argument('-observe', action='store_true', 
		help="(bool) Animate flag to true")
	parser.add_argument('-log', action='store_true', 
		help="(bool) Animate flag to true")
	args = parser.parse_args()

	# Simulation parameters
	fitness, controller, agent, pr_states, pr_actions = \
			parameters.get(args.controller)

	# Load and build the simulator
	sim = sim(savefolder="data/%s/optimization_%i/"%(controller,args.id))

	# Load the transition models
	filelist_training = [f for f in os.listdir(args.folder_training) \
											if f.endswith('.npz')]
	####################################################################


	####################################################################
	# Policy optimization

	# First we iterate over each log to build the transition model
	for j, filename in enumerate(sorted(filelist_training)):
		# The first time, set up the model, then just update it
		if j == 0:
			sim.load(args.folder_training+filename, verbose=False)
		else:
			sim.load_update(args.folder_training+filename)

	# Now we optimize the policy

	## Settings for re-iterations
	settings = {"time_limit":args.t,
		"robots":args.n,
		"environment":"square20",
		"policy_filename":"conf/policies/temp.txt", 
		"pr_states":pr_states,
		"pr_actions":pr_actions,
		"run_id":args.id,
		"fitness":fitness,
		"controller":controller,
		"agent":agent}

	## Initial policy to optimize from
	policy = np.random.rand(pr_states,pr_actions)

	## Optimization
	sim.disp()
	policy = sim.optimize(policy, iterations=args.iterations, settings=settings)
	####################################################################


	####################################################################
	# Evaluate the results

	## This is done either visually or by benchmarking 
	## with many simulations (default=benchmark)
	## depending on the input arguments
	## use -observe True to just run the result with the animation turned on
	## If not set, the solution will be evaluated 100 times

	if args.observe:
		# Run simulation with animation=ON so that you 
		# can see what's happening in a sample run	
		## Build
		sim.make(controller, agent, animation=True, verbose=False, logger=False)
		
		## Run
		settings["time_limit"] = 0 # Infinite time
		settings["policy_filename"] = sim.save_policy(policy, pr_actions)
		sim.run(**settings)

	elif args.log:
		# Build the simulator with the desired settings
		sim.make(controller, agent, animation=args.animate, verbose=False, logger=True)

		## Run
		settings["policy_filename"] = sim.save_policy(policy, pr_actions)	
		sim.run(**settings)

		# Save the log files
		sim.save_log(filename_ext="sample_log")

	else:	
		# Run a benchmark
		f = sim.benchmark(policy, make=True, **settings)
		
		# Save all received fitnesses
		fh.save_pkl(f,"data/%s/benchmark_optimized_%s_t%i_r%i_runs%i_id%i.pkl"
			%(controller,controller,args.t,args.n,args.runs,args.id))
	####################################################################