import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from classes import desired_states_extractor as dse
import numpy as np
from tools import fileHandler as fh
from tqdm import tqdm
import argparse
import os
fig = plt.figure()
import torch 

parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('folder_training', type=str, help="(str) Training data folder", default=None)
parser.add_argument('file', type=str, help="(str) Relative path to npz log file used for analysis", default=None)
args = parser.parse_args()

dse = dse.desired_states_extractor()
filelist_training = [f for f in os.listdir(args.folder_training) if f.endswith('.npz')]

for i, filename in enumerate(filelist_training):
	model,loss_history = dse.train(args.folder_training+filename) # Train the model
	print("Loading validation set")
	tc, sc, fc = dse.extract_states(args.file)
	print("Evaluating against validation set")
	error, corr, y_pred = dse.evaluate_model(model, sc, fc)
	print(corr)
	plt.plot(tc,fc);
	plt.plot(tc,y_pred);
	plt.xlabel("Time [s]")
	plt.ylabel("Fitness $F_g$")
	plt.gcf().subplots_adjust(bottom=0.15)
	plt.draw()
	plt.pause(0.001)
	plt.clf()
	