#!/usr/bin/env python3
"""
Plot the evolutions in a folder
@author: Mario Coppola, 2020
"""
from tools import fileHandler as fh
import argparse, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tools import prettyplot as pp
from classes import evolution

import matplotlib.pyplot as plt
matplotlib.rc('text', usetex=True)

# Input argument parser
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('controller', type=str, help="(str) file", default=None)
parser.add_argument('folder', type=str, help="(str) file", default=None)
parser.add_argument('-format', type=str, help="(str) Controller", default="pdf")

args = parser.parse_args()

from simulators import swarmulator

s = swarmulator.swarmulator(verbose=False)

filelist = [f for f in os.listdir(args.folder) if f.startswith('sample_log')] # Log file list
plt = None # Initialize

# Plot all
for file in filelist:
    plt = s.plot_log_column(file=args.folder+file,column=5,colname="Fitness [-]", show=False, plot=plt)
plt = pp.adjust(plt)

# Save or show
fname = "fitness_logs_%s.%s"%(args.controller,args.format)
if fname is not None:
	folder = "figures/fitness_logs/"
	if not os.path.exists(os.path.dirname(folder)): os.makedirs(os.path.dirname(folder))
	filename_raw = os.path.splitext(os.path.basename(fname))[0]
	plt.savefig(folder+"%s.%s"%(filename_raw,args.format))
else:
	plt.show()