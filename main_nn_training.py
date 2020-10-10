#!/usr/bin/env python3
"""
This file uses a spacified training data set to 
	1) train a model based on the dataset in the folder "folder_training" (argument #1)
	2) validate the model based on the dataset in the folder "folder_validation" (arguemnt #2)
The results are saved to the folder "savefolder (argument 3)

The function takes in relative arguments as above.

@author: Mario Coppola, 2020
"""

import pickle, os, argparse, copy
from tqdm import tqdm
import numpy as np
from classes import simulator, desired_states_extractor
from tools import fileHandler as fh
from tools import matrixOperations as matop

if __name__=="__main__":

	####################################################################
	# Initialize
	# Input arguments
	parser = argparse.ArgumentParser(description=
		'Simulate a task to gather the data for optimization'
	)
	parser.add_argument('folder_train', type=str, 
		help="(str) Training data folder", default=None)
	parser.add_argument('folder_test', type=str, 
		help="(str) Validation data folder", default=None)
	parser.add_argument('savefolder', type=str, 
		help="(str) Save folder", default=None)
	parser.add_argument('-id',  type=int, 
		help="Model ID (for save/load)", default=np.random.randint(0,10000))
	parser.add_argument('-train', action='store_true', 
		help="(bool) Train flag to true")
	parser.add_argument('-validate', action='store_true', 
		help="(bool) Validate flag to true (checks all models)")
	parser.add_argument('-evaluate', action='store_true', 
		help="(bool) Evaluate flag to true (checks last model only)")
	parser.add_argument('-layer_size', type=int, 
		help="Nodes in hidden layers", default=100)
	parser.add_argument('-layers', type=int, 
		help="Number of hiddent layers", default=3)
	parser.add_argument('-lr', type=float, 
		help="Number of hiddent layers", default=1e-5)

	args = parser.parse_args()

	# Load files
	folder_train = args.folder_train
	folder_test = args.folder_test
	save_folder = args.savefolder

	# Make the save_folder if it does not exist
	if not os.path.exists(os.path.dirname(save_folder)):
		os.makedirs(os.path.dirname(save_folder))

	files_train = [f for f in os.listdir(folder_train) if f.endswith('.npz')]
	files_test = [f for f in os.listdir(folder_test+"/") if f.endswith('.npz')]

	# Initialize desired states extractor
	dse = desired_states_extractor.desired_states_extractor()
	####################################################################


	####################################################################
	# if -train
	# Else try to load pre-trained sets in a file called "models.pkl"
	if args.train:
		nets = []
		i = 0
		for filename in tqdm(sorted(files_train)):
			model = dse.train(folder_train + filename, 
						layer_size=args.layer_size,
						layers=args.layers,
						lr=args.lr)
			print(model[0].network)
			nets.append(copy.deepcopy(model))
			i += 1
		fh.save_pkl(nets,"%s/models.pkl"%(save_folder))
	else:
		nets = fh.load_pkl("%s/models.pkl"%(save_folder))
	####################################################################


	####################################################################
	# If -validate
	# Crosscheck all models against all validation files
	if args.validate:
		v = []
		for model in tqdm(nets):
			e = []
			for filename in sorted(files_test):
				_, s, f = dse.extract_states(folder_test+"/"+filename, load_pkl=True)
				_, corr, _ = dse.evaluate_model(model[0], s, f)
				e.append(corr)
			v.append(e)
			print(np.mean(e)) # Display progress

		# Save to file
		vname = os.path.basename(os.path.dirname(folder_test))
		fh.save_pkl(v,"%s/validation_%s_id%i.pkl"%(save_folder,vname,args.id))
	####################################################################


	####################################################################
	# If -evaluate
	# Crosscheck the correlation of the last model against validation set
	# This is mainly for debugging purposes on the last model
	if args.evaluate:
		# Get the most recent network
		model = nets[-1]

		# Evaluate the correlation for the most recent network
		# to the validation dataset
		e = []
		for filename in sorted(files_test):
			_, s, f = dse.extract_states(
							args.folder_test+"/"+filename,
							load_pkl=True)
			_, corr, _ = dse.evaluate_model(model[0], s, f)
			e.append(corr)
		
		# Display some data
		print(np.mean(e)) # Mean error
		print(model[0].optimizer) # Optimizer parameters
		print(model[0].network) # Network parameters
	####################################################################