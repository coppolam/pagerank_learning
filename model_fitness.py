#!/usr/bin/env python3
"""
File to extract desired states
@author: Mario Coppola, 2020
"""

import pickle, sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from classes import desired_states_extractor
from tools import fileHandler as fh

# Input argument parser
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('file', type=str, help="(str) Relative path to npz log file used for analysis", default=None)
parser.add_argument('-evaluation', type=str, help="(str) Relative path to npz log file used for evaluation, default = None", default=None)
parser.add_argument('-load', type=bool, help="(bool) If True, attemps to load the pre-processed pkl alternative log file, which saves time, default = True", default=True)
args = parser.parse_args()
filename_raw = os.path.splitext(args.file)[0]

# Read/pre-process the log file
print("Reading log")
extractor = desired_states_extractor.desired_states_extractor()
t, s, f = extractor.extract_states(args.file,pkl=args.load)

# Make the NN model
print("Making the NN model")
model,loss_history = extractor.make_model(s, f)

# Print figure of learning performance
plt.figure(figsize=(6,3))
plt.plot(t,loss_history);
plt.xlabel("Time [s]")
plt.ylabel("Error [-]")
plt.gcf().subplots_adjust(bottom=0.15)
folder = os.path.dirname(args.file) + "/figures/"
if not os.path.exists(os.path.dirname(folder)): os.makedirs(os.path.dirname(folder))
plt.savefig(folder+"%s.pdf"%os.path.basename(filename_raw))
plt.clf()

# Evaluate against validation dataset, if indicated
if args.evaluation is not None:
	print("Evaluating against validation set")
	tc, sc, fc = extractor.extract_states(args.evaluation,args.load)
	error = extractor.evaluate_model(model, sc, fc)

	# Plot figure of error over time in validation dataset
	plt.figure(figsize=(6,3))
	plt.plot(tc,error);
	plt.xlabel("Time [s]")
	plt.ylabel("Error [-]")
	plt.gcf().subplots_adjust(bottom=0.15)
	plt.savefig(folder+"%s_validation_error.pdf"%os.path.basename(filename_raw))
	plt.clf()

	# Plot figure of error distribution in validation dataset
	plt.figure(figsize=(6,3))
	plt.hist(error, bins='auto', label='Original')
	plt.xlabel("Error [-]")
	plt.ylabel("Frequency")
	plt.gcf().subplots_adjust(bottom=0.15)
	plt.savefig(folder+"%s_validation_histogram.pdf"%os.path.basename(filename_raw))
	plt.clf()

# Extract desired states
print("Extracting desired states")
e = extractor.get_des()
des = e.get_best()
e.plot_evolution(figurename=folder+"%s_evo_des.pdf"%os.path.basename(filename_raw))

# Print output
print(des)