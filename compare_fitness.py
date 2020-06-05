#!/usr/bin/env python3
"""
File to extract desired states
@author: Mario Coppola, 2020
"""

import pickle, sys, os, argparse
import numpy as np
import desired_states_extractor
import matplotlib.pyplot as plt

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
	plt.plot(range(len(loss_history)),loss_history); plt.show()

	if args.evaluation is not None:
		print("Evaluating against validation set")
		sc, fc = extractor.extract_states(args.evaluation)
		extractor.evaluate_model(model, sc, fc)

	print("Extractig desired states")
	des = extractor.get_des()
	print(des)