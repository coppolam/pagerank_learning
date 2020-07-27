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

def plot_evolution(plt,e):
	'''Plot the evolution outcome'''
	plt.plot(range(1, len(e.stats)+1), [ s['mu'] for s in e.stats ])
	plt.fill_between(range(1, len(e.stats)+1),
				[ s['mu']-s['std'] for s in e.stats ],
				[ s['mu']+s['std'] for s in e.stats ], alpha=0.2)
	plt = pp.adjust(plt)
	return plt

# Input argument parser
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('folder', type=str, help="(str) folder", default=None)
args = parser.parse_args()

filelist = [f for f in os.listdir(args.folder) if f.startswith('evolution')]

# Load evolution API
e = evolution.evolution()

plt = pp.setup()
cl = 0
for file in filelist:
	e.load(args.folder+file)
	plt = plot_evolution(plt,e)
	if len(e.stats) > cl: plt.xlim(0,len(e.stats)); cl = len(e.stats)

plt.xlabel('Iterations')
plt.ylabel('Fitness')
# plt.legend()

# Save or show
fname = filelist[0]
if fname is not None:
	folder = "figures/"
	if not os.path.exists(os.path.dirname(folder)): os.makedirs(os.path.dirname(folder))
	filename_raw = os.path.splitext(os.path.basename(fname))[0]
	plt.savefig(folder+"%s.png"%filename_raw)
else:
	plt.show()