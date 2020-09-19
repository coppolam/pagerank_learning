#!/usr/bin/env python3
"""
Loop the simulator and gather data on the behavior
@author: Mario Coppola, 2020
"""

import os, argparse
import numpy as np
from tqdm import tqdm
from classes import simulator
from tools import fileHandler as fh
from tools import matrixOperations as matop
import parameters

if __name__=="__main__":

	####################################################################
	# Initialize

	# Arguments
	parser = argparse.ArgumentParser(
		description='Simulate a task to gather the data for optimization'
	)

	parser.add_argument('controller', type=str, 
		help="(str) Controller to use during evaluation")
	parser.add_argument('-t', type=int, default=200, 
		help="(int) Simulation time during benchmark, default = 200s")
	parser.add_argument('-n', type=int, default=30, 
		help="(int) Max size of swarm, default = 30")
	parser.add_argument('-id', type=int, default=np.random.randint(1000), 
		help="(int) ID of run, default = random")
	parser.add_argument('-runs', type=int, default=500, 
		help="(int) Number of runs")
	parser.add_argument('-environment', type=str, default="square20", 
		help="(str) Controller to use during evaluation")
	parser.add_argument('-animate', action='store_true', 
		help="(bool) Animate flag to true")

	args = parser.parse_args()

	# Simulation parameters
	fitness, controller, agent, pr_states, pr_actions = \
			parameters.get(args.controller)
	
	# Initialize and build the simulator
	sim = simulator.simulator(
			savefolder="data/%s/data_%i/"%(controller,args.id))
	sim.make(controller, agent, animation=args.animate, verbose=False)
	####################################################################


	####################################################################
	# Run the data generation

	for i in tqdm(range(args.runs)):	
		# Generate a random policy
		policy = np.random.rand(pr_states,pr_actions)

		# Save a temp policy
		policy_filename = sim.save_policy(policy,pr_actions)

		# Run
		sim.run(time_limit=args.t, 
			robots=np.random.randint(1,args.n), 
			environment=args.environment, 
			policy_filename=policy_filename, 
			pr_states=pr_states, 
			pr_actions=pr_actions, 
			run_id=args.id, 
			fitness=fitness)

		# Save files
		filename_ext = "%s_t%i_r%i_id%s_%i" % \
			(args.controller, args.t, args.n, sim.run_id, i)
		sim.save_learning_data(filename_ext=filename_ext)
	
	####################################################################