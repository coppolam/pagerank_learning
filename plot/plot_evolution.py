#!/usr/bin/env python3
"""
Plot the evolutions in a folder
@author: Mario Coppola, 2020
"""
from tools import fileHandler as fh
import argparse, os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tools import prettyplot as pp
from classes import evolution

import matplotlib.pyplot as plt
matplotlib.rc('text', usetex=True)

def plot_evolution(plt,e,label=None):
	'''Plot the evolution outcome'''
	if label.startswith("evolution_mbe"):
		color = "green"
		ls = '-'
		lw = 3
	else:
		color = "gray"
		ls = '--'
		lw = 1

	# Mean performance
	l = plt.semilogx(range(1, len(e.stats)+1), 
			[ s['mu'] for s in e.stats ],
			color=color,ls=ls,lw=lw)

	# Standard deviation
	# plt.fill_between(range(1, len(e.stats)+1),
	# 			[ s['mu']-s['std'] for s in e.stats ],
	# 			[ s['mu']+s['std'] for s in e.stats ], 
	# 			alpha=0.2)
	
	# Min-max performance
	# plt.fill_between(range(1, len(e.stats)+1),
	# 			[ s['min'] for s in e.stats ],
	# 			[ s['max'] for s in e.stats ], 
	# 			alpha=0.1, 
	# 			color="green")
	
	# Return
	return plt, l

# Input argument parser
def main(args):
	parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
	parser.add_argument('folder', type=str, help="(str) folder", default=None)
	parser.add_argument('-format', type=str, default="pdf")
	args = parser.parse_args(args)

	# Load evolution API
	e = evolution.evolution()
	
	# Initalize plot
	plt = pp.setup()

	# Initialize baseline limit in x axis
	lim = 0

	# Plot all evolution files found in the given folder together
	## Extract all evolution files in the indicated folder
	filelist = sorted([f for f in os.listdir(args.folder) if 
					f.startswith('evolution_') and f.endswith('pkl')])

	# filelist = [f for f in os.listdir(args.folder) if 
	# 				f.startswith('evolution_') and f.endswith('pkl')]
	li = []
	## For each file in the list, add the evolution to the plot
	for file in filelist:
    	## Load the file
		e.load(args.folder + file)

		## Plot it
		print(file)
		plt, l = plot_evolution(plt,e,label=str(file))
		li += l
		## Fix lim according to the highest number of generations
		if len(e.stats) > lim:
			lim = len(e.stats)

	# Axis labels and dimensions
	plt.xlim(0,lim)
	plt.xlabel('Generation')
	plt.ylabel('Fitness')
	plt.legend([li[0],li[-1]], ['Model-based runs','Standard runs'])
	plt = pp.adjust(plt)
	
	# Save figure
	fname = os.path.dirname(args.folder)
	folder = "figures/evolution/"
	if not os.path.exists(os.path.dirname(folder)): 
		os.makedirs(os.path.dirname(folder))
	filename_raw = os.path.splitext(os.path.basename(fname))[0]
	plt.savefig(folder+"evolution_%s.%s"%(filename_raw,args.format))
	plt.close()

if __name__ == "__main__":
	main(sys.argv[1:])