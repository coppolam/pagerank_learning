#!/usr/bin/env python3
"""
Train and validate the data gathered from the loop.py script
@author: Mario Coppola, 2020
"""

import pickle, os, argparse, copy
from tqdm import tqdm
import numpy as np
from classes import simulator, desired_states_extractor
from tools import fileHandler as fh
from tools import matrixOperations as matop

# Input arguments
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('folder_training', type=str, help="(str) Training data folder", default=None)
parser.add_argument('folder_validation', type=str, help="(str) Validation data folder", default=None)
parser.add_argument('savefolder', type=str, help="(str) Save folder", default=None)
parser.add_argument('-train', action='store_true', help="(bool) Train flag to true")
parser.add_argument('-validate', action='store_true', help="(bool) Validate flag to true")
args = parser.parse_args()

# Train the network
dse = desired_states_extractor.desired_states_extractor()
nets = []
filelist_training = [f for f in os.listdir(args.folder_training) if f.endswith('.npz')]
if args.train:
	for filename in tqdm(sorted(filelist_training)):
		model = dse.train(args.folder_training+filename)
		nets.append(copy.deepcopy(model))
	fh.save_pkl(nets,args.savefolder+"/models.pkl")
else:
	nets = fh.load_pkl(args.savefolder+"/models.pkl")

# Validate against validation set
filelist_validation = [f for f in os.listdir(args.folder_validation+"/") if f.endswith('.npz')]
v = []
if args.validate:
	for model in tqdm(nets):
		e = []
		for filename in sorted(filelist_validation): # Crosscheck against all  validatin files
			_, s, f = dse.extract_states(args.folder_validation+"/"+filename, pkl=True)
			_, corr, _ = dse.evaluate_model(model[0], s, f)
			e.append(np.mean(corr))
		v.append(e)

# Save
vname = os.path.basename(os.path.dirname(args.folder_validation))
fh.save_pkl(v,args.savefolder + "/validation_" + vname + ".pkl")