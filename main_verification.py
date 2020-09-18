import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
	import os
import networkx as nx

import parameters

from tools import fileHandler as fh
from tools import matrixOperations as matop
from tools import prettyplot as pp

from classes import pagerank_optimization as propt
from classes import simulator, evolution, desired_states_extractor, conditions

import plot_paper_model as l

# Argument parser
parser = argparse.ArgumentParser(
	description='Simulate a task to gather the data for optimization')

parser.add_argument('controller', type=str, 
	help="(str) Controller to use")
parser.add_argument('training_folder', type=str, 
	help="(str) Training folder to use")
parser.add_argument('-iterations', type=int, default=0,
	help="(int) Number of iterations")
parser.add_argument('-format', type=str, default="pdf", 
	help="(str) Training folder to use")
parser.add_argument('-plot', action='store_true', 
	help="(bool) Animate flag to true")
parser.add_argument('-verbose', action='store_true', 
	help="(bool) Animate flag to true")
parser.add_argument('-t', type=int, default=200,
	help="(int) Simulation time during benchmark, default = 200s")
parser.add_argument('-n', type=int, default=30,
	help="(int) Size of swarm, default = 30")
parser.add_argument('-id', type=int, default=np.random.randint(1000),
	help="(int) ID of run, default = random")

args = parser.parse_args()

# Load parameters
fitness, controller, agent, pr_states, pr_actions = parameters.get(args.controller)

# Load simulator model
sim = simulator.simulator()
sim = l.learn_model(sim, args.training_folder, discount=1.0)
sim.make(controller, agent, verbose=False) # Build

# Get desired states
dse = desired_states_extractor.desired_states_extractor()
dse.load_model("data/%s/models.pkl"%controller,modelnumber=499)
des = dse.get_des(dim=pr_states, gens=100, popsize=100)

# Optimize policy
policy = np.random.rand(pr_states,pr_actions)
policy = sim.optimize(policy, des)

# Perform additional iterations, if you want (not done by default)
i = 0
while i < args.iterations:
	print("\nIteration %i\n"%i)
	policy_filename = sim.save_policy(sim, policy, pr_actions)

	# Run simulation
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

# Load up sim variables locally
A = sim.A
E = sim.E
H0 = np.sum(A, axis=0)
with np.errstate(divide='ignore',invalid='ignore'):
	r = H0.sum(axis=1) / E.sum(axis=1)
	r = np.nan_to_num(r) # Remove NaN Just in case
	alpha = r / (1 + r)

# Get optimized policy
H1 = propt.update_H(A, policy)

if args.plot:

	# Calculate Google matrices
	G0 = np.diag(alpha).dot(H0) + np.diag(1-alpha).dot(E)
	G1 = np.diag(alpha).dot(H1) + np.diag(1-alpha).dot(E)

	# PageRank scores
	prH0 = matop.pagerank(H0)
	prE = matop.pagerank(E)
	pr0 = matop.pagerank(G0)
	pr1 = matop.pagerank(G1)

	# Get original fitness and new fitness
	f0 = propt.pagerankfitness(pr0, des)
	f1 = propt.pagerankfitness(pr1, des)

	# Make a folder to store the figures
	folder = "figures/pagerank"
	if not os.path.exists(os.path.dirname(folder)):
		os.makedirs(os.path.dirname(folder))
		
	# FIGURE: Plot pagerank
	plt = pp.setup()
	plt.bar(np.array(range(prH0[0].size)),prH0[0],
		alpha=0.5,
		label="$PR^\pi$, $\mathbf{H^\pi}$ only")
	plt.bar(np.array(range(prE[0].size)),prE[0],
		alpha=0.5,
		label="$PR^\pi$, $\mathbf{E}$ only")
	plt = pp.adjust(plt)
	plt.xlabel("State")
	plt.ylabel("PageRank [-]")
	plt.legend()
	plt.savefig("%s/pagerank_original_%s.%s"%(folder,controller,args.format))

	# FIGURE: Diff plot of pagerank values
	plt = pp.setup()
	c = ["blue","green"]
	color_list = list(map(lambda x: c[1] if x > 0.01 else c[0],des))
	if controller == "forage":
		plt.bar(range(pr1[0].size),(pr1[0]-pr0[0])*1000,
			label="$PR^\pi-PR^{\pi^\star}$",
			color=color_list)
		plt.ylabel("$\Delta$ PageRank ("r"$\times$"r"1000) [-]")
	else:
		plt.bar(range(pr1[0].size),(pr1[0]-pr0[0]),
			label="$PR^\pi-PR^{\pi^\star}$",
			color=color_list)
		plt.ylabel("$\Delta$ PageRank [-]")
	plt = pp.adjust(plt)
	plt.xlabel("State [-]")
	custom_lines = [matplotlib.lines.Line2D([0], [0], color="blue", lw=20),
                matplotlib.lines.Line2D([0], [0], color="green", lw=20)]
	plt.legend(custom_lines,['Transitional', 'Desired'])
	plt.savefig("%s/pagerank_diff_%s.%s"%(folder,controller,args.format))

	# FIGURE: Plot of absolute pagerank values
	plt = pp.setup()
	plt.bar(range(pr0[0].size),pr0[0],
		alpha=0.5,
		label="$PR^\pi$")
	plt.bar(range(pr1[0].size),pr1[0],
		alpha=0.5,
		label="$PR^{\pi^\star}$")
	plt.xlabel("State [-]")
	plt.ylabel("PageRank [-]")
	plt = pp.adjust(plt)
	plt.legend()
	plt.savefig("%s/pagerank_new_%s.%s"%(folder,controller,args.format))

if args.verbose:
	print("\n------- MODEL -------\n")
	print("\nH0 matrix:\n", H0)
	print("\nH1 matrix:\n", H1)
	print("\nE matrix:\n", E)
	print("\nalpha vector:\n", alpha)
	print("\n------- POLICY -------\n", policy)
	print("\n------- STATS -------\n")
	print("Original fitness =", f0[0])
	print("New fitness =", f1[0])

# Check conditions
c = conditions.verification(H0,H1,E,policy,des)
c.verify()

# Save
np.savez("data/%s/optimization.npz"%controller,
	H0=H0,H1=H1,E=E,policy=policy,des=des)