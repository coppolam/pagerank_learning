#!/usr/bin/env python3
"""
Learn the model
@author: Mario Coppola, 2020
"""

import os, argparse, sys
import matplotlib.pyplot as plt
import numpy as np
from classes import simulator
from tools import matrixOperations as matop
from tools import prettyplot as pp

sim = simulator.simulator()
	
def load_filelist(f):
	'''Loads all npz files from the specified folder'''
	return sorted([f for f in os.listdir(f) if f.endswith('.npz')])

def evaluate_model_values(f, a=0):
	# Get all the files
	filelist = load_filelist(f)

	# Load a transition model
	v = []
	for j, filename in enumerate(filelist):
		sim.load(f+filename,verbose=False)
		if j == 0:
			m = sim.A[a]
		else:
			m += sim.A[a]
		v.append(matop.normalize_rows(m).flatten())
	
	data = np.array(v).T
	
	return data

def learn_model(sim, f):
	filelist = load_filelist(f)

	v = []
	for j, filename in enumerate(filelist):
		if j == 0:
			sim.load(f+filename,verbose=False)
		else:
			sim.load_update(f+filename,verbose=False)
	
	return sim.A

def main(args):
    # Parse arguments
	
	## Load parser
	parser = argparse.ArgumentParser(
		description='Simulate a task to gather the data for optimization'
	)

	## Main arguments
	parser.add_argument('controller', type=str, 
		help="(str) Training data folder")
	parser.add_argument('folder_training', type=str, 
		help="(str) Training data folder")
	parser.add_argument('-format', type=str, default="pdf", 
		help="(str) Training data folder")
	
	## Parse
	args = parser.parse_args(args)

	# Load data
	data = evaluate_model_values(args.folder_training)
	
	# Set folder
	folder = "figures/model/"

	# Plot the difference to the last estimate to check the convergence
	plt = pp.setup()
	for d in data:
    
		# Only plot the ones that showed transitions
		if d[-1] > 0:
			plt.plot(d-(d[-1])) # Residual
	
	plt.xlabel("Simulation number")
	plt.ylabel("Difference to final estimate")
	plt = pp.adjust(plt)
	plt.ylim((-1,1))

	# Create a directory if it doesn't exist
	if not os.path.exists(os.path.dirname(folder)): 
		os.makedirs(os.path.dirname(folder))
	vname = os.path.basename(os.path.dirname(args.folder_training))

	# Save the figure
	plt.savefig(folder+"model_%s_%s.%s"%(args.controller,vname,args.format))
	plt.close()
	
if __name__ == "__main__":
	main(sys.argv[1:])