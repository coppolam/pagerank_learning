#!/usr/bin/env python3
"""
Plot the evolutions in a folder
@author: Mario Coppola, 2020
"""
import argparse, os, sys
import numpy as np
import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.rc('text', usetex=True) # Allow Latex text

from tools import swarmulator
from tools import prettyplot as pp
from tools import fileHandler as fh
from classes import evolution
import matplotlib.pyplot as plt

def transpose_list(data):
	'''List comprehension to rotate a 2D list object'''
	return [[row[i] for row in data] for i in range(len(data[0]))]

def process_logs(folder,startswith):
	# Swarmulator API (to extract the log files)
	s = swarmulator.swarmulator(verbose=False)

	# Get all data
	tv = [] # Time vectors
	fv = [] # Fitness vectors
	## Log file list optimized
	filelist = [f for f in os.listdir(folder) 
					if f.startswith(startswith)]

	for file in filelist:
		# Load the data from a log file, column=5 is the fitness
		t,f = s.load_column(file=folder+file, column=5)
		tv.append(t)
		fv.append(f)

	# Align values and transpose
	fv = list(map(lambda t,x: np.interp(tv[0],t,x),tv,fv))
	fv = transpose_list(fv)

	# Get the mean and the standard deviation at each recorded time step
	f_mean = np.mean(fv,1)
	f_std = np.std(fv,1)

	return tv, fv, f_mean, f_std, len(filelist)


def main(args):
	# Input argument parser
	parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
	parser.add_argument('controller', type=str, help="(str) path to logs folder")
	parser.add_argument('-format', type=str, default="pdf")
	args = parser.parse_args(args)

	pf = "data/%s/"%args.controller
	folders = ["onlinelearning_detach_false_shared_false_iter_10_boost_1/"\
	,"onlinelearning_detach_false_shared_true_iter_10_boost_1/"]
	plt = pp.setup()
	li = []
	for i,f in enumerate(folders):
		tv, fv, f_mean, f_std, nlogs = process_logs(pf+f, "log")
		
		if i == 0: c = "red"
		else: c = "blue"
		
		# Plot
		l = plt.plot(tv[0], fv, alpha=0.2,color="gray")
		l2 = plt.plot(tv[0], f_mean, color=c)
		li += l2
		plt.fill_between(tv[0],
			f_mean-f_std,
			f_mean+f_std,
			alpha=0.2,
			color=c)
		

	# Axis
	# plt.legend()
	plt.legend([li[0],li[-1]], ['Local model','Shared local model'])
	plt = pp.adjust(plt)
	plt.xlabel("Time [s]")
	plt.ylabel("Fitness [-]")

	# # Save or show
	fname = "fitness_logs_%s.%s"%(args.controller,args.format)
	folder = "figures/onlinelearning/"
	if not os.path.exists(os.path.dirname(folder)): 
		os.makedirs(os.path.dirname(folder))
	filename_raw = os.path.splitext(os.path.basename(fname))[0]
	plt.savefig(folder+"onlinelearning_%s.%s"%(filename_raw,args.format))
	plt.close()
	
	# # Show weights analysis
	# # For each experiment
	# for i in range(1,nlogs+1):
    # 	# Get all policy files
	# 	filelist = [f for f in os.listdir(args.folder) 
	# 					if f.startswith("policy_%i"%i)]

	# 	# For each agent, load the files
	# 	l = []
	# 	for file in filelist:
	# 		f = np.loadtxt(args.folder+file)
	# 		l.append(f)

	# 	plt.plot(l[0],marker=".") # move [0]
	# 	plt.show()


if __name__ == '__main__':
	main(sys.argv[1:])