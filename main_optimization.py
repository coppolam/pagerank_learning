#!/usr/bin/env python3
"""
Load the training data + model and optimize the behavior accordingly.
Then evaluate it and save the results.

@author: Mario Coppola, 2020
"""

import pickle, os, argparse
import numpy as np
from classes import simulator, evolution, desired_states_extractor
from tools import fileHandler as fh
from tools import matrixOperations as matop
from simulators import parameters

def save_policy(sim, policy, pr_actions):
	'''Save the policy in the correct format for use in Swarmulator'''
	# Resize pol
	policy = np.reshape(policy,(policy.size//pr_actions,pr_actions)) 
	
	# Normalize rows if more than one column
	if pr_actions > 1:
		policy = matop.normalize_rows(policy)
	
	# Policy filename
	policy_filename = "conf/policies/policy_learnloop.txt"
	policy_file = sim.sim.path + "/" + policy_filename
	
	if policy.shape[1] == 1:
		fh.save_to_txt(policy.T, policy_file)
	else:
		fh.save_to_txt(policy, policy_file)
	
	return policy_filename

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
fitness, controller, agent, pr_states, pr_actions = parameters.get(args.controller)

# Load and build the simulator
sim = simulator.simulator(savefolder="data/%s/learnloop_%i/"%(controller,args.id))
sim.make(controller, agent, animation=args.animate, verbose=False)

# Get the desired states using the trained feed-forward network
dse = desired_states_extractor.desired_states_extractor()
dse.load_model("data/%s/models.pkl"%controller,modelnumber=499) # 499 to use the last model
des = dse.get_des(dim=pr_states)

# Initial policy to optimize from
policy = np.random.rand(pr_states,pr_actions)

# Load the transition models
filelist_training = [f for f in os.listdir(args.folder_training) if f.endswith('.npz')]
v = []
# Iterate over each log to build the model
for j, filename in enumerate(sorted(filelist_training)):
	# The first time, set up the model, then just update it
	if j == 0:
		sim.load(args.folder_training+filename, verbose=False)
	else:
		sim.load_update(args.folder_training+filename)

# Optimize the policy
policy = sim.optimize(policy, des)

# Perform additional iterations on the policy (not done by default)
i = 0
while i < args.iterations:
	print("\nIteration %i\n"%i)
	# Run simulation
	policy_filename = save_policy(sim, policy, pr_actions)
	sim.run(time_limit=args.t, robots=args.n, environment="square20", policy=policy_filename, 
		pr_states=pr_states, pr_actions=pr_actions, run_id=args.id, fitness=fitness)

	# Save data
	logname = "%s_%s_t%i_r%i_id%s_%i"%(controller, agent, args.t, args.n, sim.run_id, i)
	logfile = sim.save_learning_data(filename_ext=logname)

	# Optimization
	des = dse.run(logfile+".npz", load=False, verbose=False) # Update desired states 
	sim.load_update(logfile+".npz",discount=1.0) # Update model
	policy = sim.optimize(policy, des) # Optimize again
	
	i += 1
	print(des)
	print(policy)
	
# Evaluate the result
## This is done either visually or by benchmarking 
## with many simulations (default=benchmark)
## depending on the input arguments
## use -observe True to just run the result with the animation turned on
## If not set, the solution will be evaluated 100 times
if args.observe:
	# Run simulation with animation=on so that you can see what's happening in a sample run
	policy_filename = save_policy(sim, policy, pr_actions)
	
	# Build
	sim.make(controller, agent, 
		animation=True, 
		verbose=False)
	
	# Run
	sim.run(time_limit=0, 
		robots=args.n, 
		environment=args.environment, 
		policy=policy_filename, 
		pr_states=pr_states, 
		pr_actions=pr_actions, 
		run_id=args.id, 
		fitness=fitness)

elif args.log:
    	
	# Set up policy
	policy_filename = save_policy(sim, policy, pr_actions)
	
	# Build the simulator with the desired settings
	sim.make(controller, agent, 
		animation=False, 
		verbose=False, 
		logger=True)
		
	# Run it
	sim.run(time_limit=args.t, 
		robots=args.n, 
		environment=args.environment, 
		policy=policy_filename, 
		pr_states=pr_states, 
		pr_actions=pr_actions, 
		run_id=args.id, 
		fitness=fitness)

	# Save the log files
	sim.save_log(filename_ext="sample_log")

else:
	
	# Set up policy
	policy = matop.normalize_rows(policy)
	
	# Run a benchmark
	f = sim.benchmark(controller, 
			agent, 
			policy, 
			fitness, 
			make=True, 
			robots=args.n, 
			runs=args.runs, 
			environment=args.environment, 
			time_limit=args.t, 
			pr_states=pr_states, 
			pr_actions=pr_actions)
	
	# Save all received fitnesses
	fh.save_pkl(f,"data/%s/benchmark_optimized_%s_t%i_r%i_runs%i_id%i.pkl"
		%(controller,controller,args.t,args.n,args.runs,args.id))
