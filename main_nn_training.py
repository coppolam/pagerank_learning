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
import torch 

# Input arguments
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('folder_training', type=str, help="(str) Training data folder", default=None)
parser.add_argument('folder_validation', type=str, help="(str) Validation data folder", default=None)
parser.add_argument('savefolder', type=str, help="(str) Save folder", default=None)
parser.add_argument('-id',  type=int, help="Model ID (for save/load)", default=1)
parser.add_argument('-train', action='store_true', help="(bool) Train flag to true")
parser.add_argument('-validate', action='store_true', help="(bool) Validate flag to true (checks all models)")
parser.add_argument('-evaluate', action='store_true', help="(bool) Evaluate flag to true (checks last model only)")
args = parser.parse_args()

#Initialize desired states extractor
dse = desired_states_extractor.desired_states_extractor()

# Train the network (or else load the model)
nets = []
filelist_training = [f for f in os.listdir(args.folder_training) if f.endswith('.npz')]
if args.train:
	i = 0
	for filename in tqdm(sorted(filelist_training)):
		model = dse.train(args.folder_training+filename)
		nets.append(copy.deepcopy(model))
		i += 1
	fh.save_pkl(nets,"%s/models.pkl"%(args.savefolder))
else:
	nets = fh.load_pkl("%s/models.pkl"%(args.savefolder))

# Validate against validation set
filelist_validation = [f for f in os.listdir(args.folder_validation+"/") if f.endswith('.npz')]

# Crosscheck the correlation of the last model against validation set
if args.evaluate:
	model = nets[-1]
	e = []
	for filename in sorted(filelist_validation):
		_, s, f = dse.extract_states(args.folder_validation+"/"+filename, pkl=True)
		_, corr, _ = dse.evaluate_model(model[0], s, f)
		e.append(np.mean(corr))
	print(np.mean(e)) # visualize
	print(model[0].optimizer)
	print(model[0].network)

# Crosscheck all models against all validation files
v = []
if args.validate:
	for model in tqdm(nets):
		e = []
		for filename in sorted(filelist_validation):
			_, s, f = dse.extract_states(args.folder_validation+"/"+filename, pkl=True)
			_, corr, _ = dse.evaluate_model(model[0], s, f)
			e.append(np.mean(corr))
		v.append(e)
		print(np.mean(e)) # visualize

# Save to file
vname = os.path.basename(os.path.dirname(args.folder_validation))
fh.save_pkl(v,"%s/validation_%s_id%i.pkl"%(args.savefolder,vname,args.id))