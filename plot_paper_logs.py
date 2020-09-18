#!/usr/bin/env python3
"""
Plot the evolutions in a folder
@author: Mario Coppola, 2020
"""
import argparse, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('text', usetex=True) # Allow Latex text

from tools import swarmulator
from tools import prettyplot as pp
from tools import fileHandler as fh
from classes import evolution

def transpose_list(data):
	'''List comprehension to rotate a 2D list object'''
	return [[row[i] for row in data] for i in range(len(data[0]))]

# Input argument parser
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('controller', type=str, help="(str) controller", default=None)
parser.add_argument('folder', type=str, help="(str) path to logs folder", default=None)
parser.add_argument('-format', type=str, help="(str) plot format, default=pdf", default="pdf")
args = parser.parse_args()

# Swarmulator API (to extract the log files)
s = swarmulator.swarmulator(verbose=False)

# Get all data
tv = [] # Time vectors
fv = [] # Fitness vectors
filelist = [f for f in os.listdir(args.folder) if f.startswith('sample_log')] # Log file list
for file in filelist:
	# Load the data from a log file, column=5 is the fitness
	t,f = s.load_column(file=args.folder+file,column=5,colname="Fitness [-]", show=False, plot=plt)
	tv.append(t)
	fv.append(f)

# Align it and transpose it
fv = list(map(lambda t,x: np.interp(tv[0],t,x),tv,fv))
fv = transpose_list(fv)

# Get the mean and the standard deviation at each recorded time step
f_mean = np.mean(fv,1)
f_std = np.std(fv,1)

# Plot
plt = pp.setup()
plt.plot(tv[0], f_mean)
plt.fill_between(tv[0],
	f_mean-f_std,
	f_mean+f_std,
	alpha=0.2,
	color="blue")
plt = pp.adjust(plt)
plt.xlabel("Time [s]")
plt.ylabel("Fitness [-]")

# Save or show
fname = "fitness_logs_%s.%s"%(args.controller,args.format)
if fname is not None:
	folder = "figures/fitness_logs/"
	if not os.path.exists(os.path.dirname(folder)): os.makedirs(os.path.dirname(folder))
	filename_raw = os.path.splitext(os.path.basename(fname))[0]
	plt.savefig(folder+"%s.%s"%(filename_raw,args.format))
else:
	plt.show()