#!/usr/bin/env python3
"""
Loop the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""

import pickle, sys, matplotlib, os, argparse, copy
from tqdm import tqdm
import numpy as np
from classes import simulator, desired_states_extractor
from tools import fileHandler as fh
from tools import matrixOperations as matop
import matplotlib.pyplot as plt
matplotlib.rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('folder_training', type=str, help="(str) Training data folder")
parser.add_argument('-debug', type=bool, default=False, help="(bool) Debug (only 10 iterations), default False")
args = parser.parse_args()

sim = simulator.simulator()
filelist_training = [f for f in os.listdir(args.folder_training) if f.endswith('.npz')]

if args.debug: i = 0
v = []
for j, filename in tqdm(enumerate(sorted(filelist_training))):
	sim.load(args.folder_training+filename,verbose=False)
	if j == 0: m = sim.A[0]
	else: m += sim.A[0]
	v.append(matop.normalize_rows(m).flatten())
	if args.debug:
		i += 1
		if i > 20: break


data = np.array(v).T

for d in data: plt.plot(d)
plt.show()

print("Done")