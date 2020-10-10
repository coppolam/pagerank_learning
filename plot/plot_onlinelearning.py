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

# Input argument parser
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('file', type=str, help="(str) controller")
args = parser.parse_args()

# Swarmulator API (to extract the log files)
s = swarmulator.swarmulator(verbose=False)

# Load the data from a log file, column=5 is the fitness
s.plot_log_column(file=args.file,
    column=5,colname="Fitness [-]", show=True, plot=plt)