#!/usr/bin/env python3
"""
Learn the model
@author: Mario Coppola, 2020
"""

import matplotlib, os, argparse
from tqdm import tqdm
import numpy as np
from classes import simulator
from tools import matrixOperations as matop
import matplotlib.pyplot as plt
from tools import prettyplot

def evaluate_model_values(f,a=0):
	sim = simulator.simulator()
	filelist = [f for f in os.listdir(f) if f.endswith('.npz')]
	v = []
	for j, filename in tqdm(enumerate(sorted(filelist))):
		sim.load(f+filename,verbose=False)
		if j == 0: m = sim.A[a]
		else: m += sim.A[a]
		v.append(matop.normalize_rows(m).flatten())
	data = np.array(v).T
	return data

def learn_model(sim,f,discount=1.0):
	filelist = [f for f in os.listdir(f) if f.endswith('.npz')]
	v = []
	for j, filename in tqdm(enumerate(sorted(filelist))):
		if j == 0: sim.load(f+filename,verbose=False)
		else: sim.load_update(f+filename,discount=discount,verbose=False)
	return sim

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
	parser.add_argument('folder_training', type=str, help="(str) Training data folder")
	args = parser.parse_args()

	data = learn_model(args.folder_training)
	for d in data:
		plt.plot(d)
	plt.show()