import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

import parameters
import conditions
import plot_paper_model as l

from tools import matrixOperations as matop
from tools import prettyplot as pp
from classes import pagerank_optimization as propt
from classes import simulator, evolution, desired_states_extractor

# Args
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('controller', type=str, help="Controller to use")
parser.add_argument('training_folder', type=str, help="Training folder to use")
parser.add_argument('-format', type=str, default="pdf", help="Training folder to use")
parser.add_argument('-plot', action='store_true', help="(bool) Animate flag to true")
parser.add_argument('-verbose', action='store_true', help="(bool) Animate flag to true")
args = parser.parse_args()

# Load parameters
fitness, controller, agent, pr_states, pr_actions = parameters.get(args.controller)

# Load simulator model
sim = simulator.simulator()
sim = l.learn_model(sim, args.training_folder, discount=1.0)

# Get desired states
dse = desired_states_extractor.desired_states_extractor()
dse.load_model("data/%s/models.pkl"%controller,modelnumber=499)
des = dse.get_des(dim=pr_states, gens=100, popsize=100)

# Optimize policy
policy = np.random.rand(pr_states,pr_actions)
policy = sim.optimize(policy, des)

# Load up sim variables locally
A = sim.A
E = sim.E
H0 = np.sum(A, axis=0)
with np.errstate(divide='ignore',invalid='ignore'):
	r = H0.sum(axis=1) / E.sum(axis=1)
	r = np.nan_to_num(r) # Remove NaN Just in case
	alpha = r / (1 + r)

# Get new policy
H1 = propt.update_H(A, policy)

# Google matrices
G0 = np.diag(alpha).dot(H0) + np.diag(1-alpha).dot(E)
G1 = np.diag(alpha).dot(H1) + np.diag(1-alpha).dot(E)

# PageRank scores
prH0 = matop.pagerank(H0)
prH1 = matop.pagerank(H1)
prE = matop.pagerank(E)
pr0 = matop.pagerank(G0)
pr1 = matop.pagerank(G1)

# Get original fitness and new fitness
f0 = propt.pagerankfitness(pr0,des)
f1 = propt.pagerankfitness(pr1,des)

import os
folder = "figures/pagerank"
if not os.path.exists(os.path.dirname(folder)):
	os.makedirs(os.path.dirname(folder))

if args.plot:
	# Plot pagerank
	plt = pp.setup()
	plt.bar(np.array(range(prH0[0].size)),prH0[0],alpha=0.5,label="PR^\pi, H^\pi")
	plt.bar(np.array(range(prE[0].size)),prE[0],alpha=0.5,label="PR^\pi, E")
	plt = pp.adjust(plt)
	plt.xlabel("State [-]")
	plt.ylabel("PageRank [-]")
	plt.legend()
	plt.savefig("%s/pagerank_original_%s.%s"%(folder,controller,args.format))

	# Diff plot of pagerank values
	plt = pp.setup()
	plt.bar(range(pr1[0].size),pr1[0]-pr0[0],alpha=0.5,label="$PR^\pi-PR^{\pi^\star}$")
	plt = pp.adjust(plt)
	plt.xlabel("State [-]")
	plt.ylabel("PageRank [-]")
	# plt.legend() # Not needed here
	plt.savefig("%s/pagerank_diff_%s.%s"%(folder,controller,args.format))

	# Plot of absolute pagerank values
	plt = pp.setup()
	plt.bar(range(pr0[0].size),pr0[0],alpha=0.5,label="$PR^\pi$")
	plt.bar(range(pr1[0].size),pr1[0],alpha=0.5,label="$PR^{\pi^\star}$")
	plt.xlabel("State [-]")
	plt.ylabel("PageRank [-]")
	plt = pp.adjust(plt)
	plt.legend()
	plt.savefig("%s/pagerank_new_%s.%s"%(folder,controller,args.format))

if args.verbose:
	print("\n------- MODEL -------")
	print("H0 matrix:\n",H0)
	print("H1 matrix:\n",H1)
	print("\nE matrix:\n",sim.E)
	print("\nalpha vector:\n",alpha)

	print("\n------- POLICY -------")
	print(policy)

	print("\n------- STATS -------")
	print("Original fitness =",f0[0])
	print("New fitness =",f1[0])

# Check conditions
c = conditions.verification(H0,H1,E,policy,des)
c.verify()