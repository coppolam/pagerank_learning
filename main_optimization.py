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
import parameters

def save_policy(sim,policy,pr_actions):
	'''Save the policy in the correct format for use in Swarmulator'''
	policy = np.reshape(policy,(policy.size//pr_actions,pr_actions)) # Resize pol
	if pr_actions > 1: policy = matop.normalize_rows(policy) # Normalize rows
	policy_filename = "conf/state_action_matrices/aggregation_policy_learnloop.txt"
	policy_file = sim.sim.path + "/" + policy_filename
	if policy.shape[1] == 1: fh.save_to_txt(policy.T, policy_file) # Number of columns = 1
	else: fh.save_to_txt(policy, policy_file)
	return policy_filename

# Input argument parser
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('controller', type=str, help="(str) Controller to use during evaluation")
parser.add_argument('folder_training', type=str, help="(str) Controller to use during evaluation")
parser.add_argument('-t', type=int, help="(int) Simulation time during benchmark, default = 200s", default=200)
parser.add_argument('-n', type=int, help="(int) Size of swarm, default = 30", default=30)
parser.add_argument('-runs', type=int, help="(int) Evaluation runs, default = 100", default=100)
parser.add_argument('-id', type=int, help="(int) ID of run, default = 1", default=1)
parser.add_argument('-animate', type=bool, help="(bool) If True, does not do a benchmark but only shows a swarm with the optimized controller, default = False", default=False)
parser.add_argument('-iterations', type=int, help="(int) Number of iterations", default=1)
args = parser.parse_args()

# Simulation parameters
fitness, controller, agent, pr_states, pr_actions = parameters.get(args.controller)

# Load and build
sim = simulator.simulator(savefolder="data/%s/learnloop_%i/"%(controller,args.id)) # Load swarmulator API
sim.make(controller, agent, animation=args.animate, verbose=False) # Build

# Desired states
dse = desired_states_extractor.desired_states_extractor()
dse.load_model("data/%s/models.pkl"%controller)
des = dse.get_des(dim=pr_states)

# Initial policy to optimize from
policy = np.random.rand(pr_states,pr_actions)

# Load model
filelist_training = [f for f in os.listdir(args.folder_training) if f.endswith('.npz')]
v = []
for j, filename in enumerate(sorted(filelist_training)):
	if j == 0: sim.load(args.folder_training+filename,verbose=False)
	else: sim.load_update(args.folder_training+filename)

# Optimize
policy = sim.optimize(policy, des)

print(des)
print(policy)

# Perform additional iterations
i = 0
while i < args.iterations:
	print("Iteration %i"%i)
	policy_filename = save_policy(sim, policy, pr_actions)

	# Run simulation
	sim.run(time_limit=args.t, robots=args.n, environment="square", policy=policy_filename, 
		pr_states=pr_states, pr_actions=pr_actions, run_id=args.id, fitness=fitness)

	# Save data
	logname = "%s_%s_t%i_r%i_id%s_%i"%(controller, agent, args.t, args.n, sim.run_id, i)
	logfile = sim.save_learning_data(filename_ext=logname)

	# Optimization procedure
	des = dse.run(logfile+".npz", load=False, verbose=False) # Update desired states 
	sim.load_update(logfile+".npz",discount=1.0) # Update model
	policy = sim.optimize(policy, des) # Optimize again
		
	i += 1

	print(des)
	print(policy)
	
# Benchmark final controller
print("----------Benchmarking-----------")
f = sim.benchmark(controller, agent, policy, fitness, robots=args.n, runs=args.runs, time_limit=args.t, make=True)
fh.save_pkl(f,"data/%s/benchmark_optimized_%s_t%i_r%i_runs%i.pkl"%(controller,controller,args.t,args.n,args.runs))

# Plot
import plot_paper_benchmark as b
b.benchmark("data/%s/benchmark_random_%s.pkl"%(controller,controller),new=f)