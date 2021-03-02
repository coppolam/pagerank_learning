#!/usr/bin/env python3
'''
Verify the behavior of the swarm according to the propositions.
And check pagerank scores with/without optimization.

@author: Mario Coppola, 2020
'''


import numpy as np
import argparse, os, sys
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import copy

import parameters

from tools import fileHandler as fh
from tools import matrixOperations as matop
from tools import prettyplot as pp

from classes import pagerank_evolve as propt
from classes import simulator, evolution, desired_states_extractor, verification

def main(args):
	####################################################################
	# Initialize

	# Argument parser
	parser = argparse.ArgumentParser(
		description='Simulate a task to gather the data for optimization'
	)
	parser.add_argument('controller', type=str, 
		help="(str) Controller to use")
	parser.add_argument('folder', type=str, 
		help="(str) Folder to use")
	parser.add_argument('-format', type=str, default="pdf", 
		help="(str) Save figure format")
	parser.add_argument('-plot', action='store_true', 
		help="(bool) Animate flag to true")
	parser.add_argument('-verbose', action='store_true', 
		help="(bool) Animate flag to true")
	args = parser.parse_args(args)

	# Load parameters
	fitness, controller, agent, pr_states, pr_actions = \
		parameters.get(args.controller)
	####################################################################


	####################################################################
	# Load optimization files
	files_train = [f for f in os.listdir(args.folder) \
		if f.startswith("optimization") and f.endswith('.npz')]

	# Unpack last file
	data = np.load(args.folder+files_train[-1])
	H0 = data["H0"].astype(float)
	H1 = data["H1"].astype(float)
	# Fix rounding errors
	H0[H0<0.01] = 0.00000
	H1[H1<0.01] = 0.00000
	E = matop.normalize_rows(data["E"])
	policy = data["policy"]
	des = data["des"]
	alpha = data["alpha"]
	####################################################################


	####################################################################
	# if -plot
	# Plot and display relevant results

	if args.plot:

		# Calculate parameters
		## Calculate Google matrices
		G0 = np.diag(alpha).dot(H0) + np.diag(1-alpha).dot(E)
		G1 = np.diag(alpha).dot(H1) + np.diag(1-alpha).dot(E)

		## PageRank scores
		prH0 = matop.pagerank(H0)
		prE = matop.pagerank(E)
		pr0 = matop.pagerank(G0)
		pr1 = matop.pagerank(G1)
		
		## Initialize pagerank optimizer for evaluation
		## Using dummy inputs, since init not needed
		p = propt.pagerank_evolve(des,np.array([H0,H1]),E) 

		## Get original fitness and new fitness
		f0 = p.pagerank_fitness(pr0, des)
		f1 = p.pagerank_fitness(pr1, des)

		# Make a folder to store the figures
		folder = "figures/pagerank"
		if not os.path.exists(os.path.dirname(folder)):
			os.makedirs(os.path.dirname(folder))
		
		#Now let's plot some figures
		import math
		xint = range(0, math.ceil(pr1[0].size),2)
		
		# Figure: Plot pagerank H and E
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
		matplotlib.pyplot.xticks(xint)
		plt.legend()
		plt.savefig("%s/pagerank_original_%s.%s" \
			%(folder,controller,args.format))
		plt.close()
		
		# Figure: Diff plot of pagerank values
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
		matplotlib.pyplot.xticks(xint)
		
		# Custom legend
		custom_lines = [
					matplotlib.lines.Line2D([0], [0], color="blue" , lw=20),
					matplotlib.lines.Line2D([0], [0], color="green", lw=20)
					]
		plt.legend(custom_lines,['Transitional', 'Desired'])
		plt.savefig("%s/pagerank_diff_%s.%s"%(folder,controller,args.format))
		plt.close()
		return
	####################################################################


	####################################################################
	# if -verbose
	# Display relevant results to terminal
	if args.verbose:
		print("\n------- MODEL -------\n")
		print("\nH0 matrix:\n", H0)
		print("\nH1 matrix:\n", H1)
		print("\nE matrix:\n", E)
		print("\nalpha vector:\n", alpha)
		print("\n------- POLICY -------\n", policy)
		# print("\n------- STATS -------\n")
		# print("Original fitness =", f0[0])
		# print("New fitness =", f1[0])

	# Check conditions on last file
	e = 0.00000001
	H0[H0>e] = 1
	H1[H1>e] = 1
	E [E >e] = 1
	H0 = H0.astype(int)
	H1 = H1.astype(int)
	E = E.astype(int)
	c = verification.verification(H0,H1,E,policy,des)
	c.verify()

	####################################################################

if __name__ == "__main__":	
	main(sys.argv[1:])