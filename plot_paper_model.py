#!/usr/bin/env python3
"""
Learn the model
@author: Mario Coppola, 2020
"""

import matplotlib, os, argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from classes import simulator
from tools import matrixOperations as matop
from tools import prettyplot as pp

sim = simulator.simulator()
	
def evaluate_model_values(f, a=0):
	filelist = [f for f in os.listdir(f) if f.endswith('.npz')]
	v = []
	for j, filename in tqdm(enumerate(sorted(filelist))):
		sim.load(f+filename,verbose=False)
		if j == 0: m = sim.A[a]
		else: m += sim.A[a]
		v.append(matop.normalize_rows(m).flatten())
	data = np.array(v).T
	return data

def learn_model(sim, f, discount=1.0):
	filelist = [f for f in os.listdir(f) if f.endswith('.npz')]
	v = []
	for j, filename in tqdm(enumerate(sorted(filelist))):
		if j == 0: sim.load(f+filename,verbose=False)
		else: sim.load_update(f+filename,discount=discount,verbose=False)
	return sim

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
	parser.add_argument('controller', type=str, help="(str) Training data folder")
	parser.add_argument('folder_training', type=str, help="(str) Training data folder")
	parser.add_argument('-format', type=str, default="pdf", help="(str) Training data folder")
	args = parser.parse_args()

	data = evaluate_model_values(args.folder_training)
	
	folder = "figures/model/"

	plt = pp.setup()
	for d in data:
		plt.plot(abs(d-d[-1])) # Residual
	plt.xlabel("Simulation")
	plt.ylabel("Residual")
	plt = pp.adjust(plt)

	if not os.path.exists(os.path.dirname(folder)): os.makedirs(os.path.dirname(folder))
	vname = os.path.basename(os.path.dirname(args.folder_training))
	
	plt.savefig(folder+"model_%s_%s.%s"%(args.controller,vname,args.format))