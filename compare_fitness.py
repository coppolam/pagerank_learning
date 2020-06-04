#!/usr/bin/env python3
"""
Simulate the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""

import pickle, sys, os, argparse
import numpy as np
import desired_states_extractor

if __name__ == "__main__":
	###########################
	#  Input argument parser  #
	###########################
	parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
	parser.add_argument('file', type=str, help="File to analyze", default=None)
	parser.add_argument('-evaluation', type=str, help="Load from file or re-run", default=None)
	args = parser.parse_args()

	extractor = desired_states_extractor.desired_states_extractor()
	s, f = extractor.extract_states(args.file)
	model = extractor.make_model(s, f)
	if args.evaluation is not None:
		sc, fc = extractor.extract_states(args.evaluation)
		extractor.evaluate_model(model, sc, fc)
	des = extractor.get_des()
	print(des)