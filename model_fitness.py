#!/usr/bin/env python3
"""
File to extract desired states
@author: Mario Coppola, 2020
"""

import pickle, sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt

from classes import desired_states_extractor

if __name__ == "__main__":
	###########################
	#  Input argument parser  #
	###########################
	parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
	parser.add_argument('file', type=str, help="File to analyze", default=None)
	parser.add_argument('-evaluation', type=str, help="Load from file or re-run", default=None)
	args = parser.parse_args()

	print("Reading log")
	extractor = desired_states_extractor.desired_states_extractor()
	s, f = extractor.extract_states(args.file)
	
	print("Making the NN model")
	model,loss_history = extractor.make_model(s, f)

	# Figure
	plt.figure(figsize=(6,3))
	plt.plot(range(len(loss_history)),loss_history);
	folder = os.path.dirname(args.file) + "/figures/"
	if not os.path.exists(os.path.dirname(folder)): os.makedirs(os.path.dirname(folder))
	plt.savefig(folder+"%s.pdf"%os.path.basename(args.file))
	plt.clf()
	
	if args.evaluation is not None:
		print("Evaluating against validation set")
		sc, fc = extractor.extract_states(args.evaluation)
		error = extractor.evaluate_model(model, sc, fc)
		plt.figure(figsize=(6,3))
		plt.plot(range(len(error)),error);
		plt.savefig(folder+"%s_validation_error.pdf"%os.path.basename(args.file))
		plt.xlabel("Time [s]")
		plt.ylabel("Error [-]")
		plt.clf()
		plt.figure(figsize=(6,3))
		plt.hist(error, bins='auto', label='Original')
		plt.xlabel("Error [-]")
		plt.ylabel("Frequency")
		plt.savefig(folder+"%s_validation_histogram.pdf"%os.path.basename(args.file))
		
	print("Extractig desired states")
	des = extractor.get_des()
	print(des)