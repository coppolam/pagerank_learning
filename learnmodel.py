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
parser.add_argument('folder_validation', type=str, help="(str) Validation data folder")
parser.add_argument('savefolder', type=str, help="(str) Save folder")
parser.add_argument('-train', type=bool, default=True, help="(bool) Train, defualt True")
parser.add_argument('-validate', type=bool, default=True, help="(bool) Validate, default True")
parser.add_argument('-debug', type=bool, default=False, help="(bool) Debug (only 10 iterations), default False")
args = parser.parse_args()

sim = simulator.simulator()
filelist_training = [f for f in os.listdir(args.folder_training) if f.endswith('.npz')]

if args.debug: i = 0
if args.train:
	for j,filename in tqdm(enumerate(sorted(filelist_training))):
		if j == 0: sim.load(args.folder_training+filename)
		else: sim.load_update(args.folder_training+filename)
		if args.debug:
			i += 1
			if i > 20: break

filelist_validation = [f for f in os.listdir(args.folder_validation+"/") if f.endswith('.npz')]
v = []
if args.debug: i = 0
if args.validate:
	for model in tqdm(nets):
		e = []
		for filename in sorted(filelist_validation): # Crosscheck against all  validatin files
			_, s, f = dse.extract_states(args.folder_validation+"/"+filename, pkl=True)
			_, corr, _ = dse.evaluate_model(model[0], s, f)
			e.append(np.mean(corr))
		v.append(e)
		if args.debug:
			i += 1
			if i > 20: break

vname = os.path.basename(os.path.dirname(args.folder_validation))
fh.save_pkl(v,args.savefolder + "/validation_" + vname + ".pkl")