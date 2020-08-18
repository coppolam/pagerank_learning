#!/usr/bin/env python3
"""
File to evaluate and plot the runtime of swarmulator
@author: Mario Coppola, 2020
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time, os
from tqdm import tqdm
import argparse
from simulators import swarmulator
matplotlib.rc('text', usetex=True)
sim = swarmulator.swarmulator(verbose=False)
	
def wall_clock_test(n,m):
	'''Test the wall clock time of single runs'''
	sim.make(clean=True,verbose=False);
	print("Wall-clock test (single simulation mode)")
	t = np.zeros([n,m])
	for i in tqdm(range(1,n+1)): # Iterate over number of robots
		for j in range(0,m): # Do multiple runs to average out
			tic = time.time()    
			f = sim.run(i) # Launch simulation run
			toc = time.time()
			t[i-1,j] = toc-tic # Runtime
	return t

def wall_clock_batch_test(n,m,batchsize):
	'''Test the wall clock time of batch (parallel) runs'''
	sim.make(clean=True,verbose=False);
	print("Wall-clock test (batch simulation mode)")
	t = np.zeros([n,m])
	for i in tqdm(range(1,n+1)): # Iterate over number of robots
		for j in range(0,m): # Do multiple runs to average out
			tic = time.time()
			f = sim.batch_run(i,batchsize) # Launch batch
			toc = time.time()
			t[i-1,j] = toc-tic # Runtime
	return t

def run(n,m,batch,filename):
	# Test wall-clock time for single runs without animation
	t = wall_clock_test(n,m)

	# Test wall-clock time for batch runs, without animation
	t_batch = wall_clock_batch_test(n,m,batch)

	return t, t_batch
	
def load(filename):
	data = np.load(filename)
	t = data['t'].astype(float)
	t_batch = data['t_batch'].astype(float)
	return t, t_batch

def plot_evaluationtime(filename,figurename=None):
	plt.clf()
	t, t_batch = load(filename)
	tmean = t.mean(axis=1)
	batchsize = 5
	alpha = 0.2
	tmean_batch = t_batch.mean(axis=1)/batchsize
	plt.plot(range(1,t.shape[0]+1),tmean,color='blue',label="Single")
	plt.plot(range(1,t_batch.shape[0]+1),tmean_batch,color='orange',label="Batch (5)")
	plt.fill_between(range(1, len(tmean)+1),
		t.min(axis=1),
		t.max(axis=1),
		color='blue', alpha=alpha)
	plt.fill_between(range(1, len(tmean)+1),
		t_batch.min(axis=1)/batchsize,
		t_batch.max(axis=1)/batchsize,
		color='orange', alpha=alpha)
	plt.xlabel("Number of robots")
	plt.ylabel("Wall-clock time [s]")
	plt.legend(loc="upper left")
	plt.savefig(figurename) if figurename is not None else plt.show()

def plot_realtimefactor(filename,tl,figurename=None):
	plt.clf()
	t, t_batch = load(filename)
	tmean = t.mean(axis=1)/tl
	batchsize = 5
	alpha = 0.2
	tmean_batch = (t_batch.mean(axis=1)/batchsize)/tl
	plt.plot(range(1,t.shape[0]+1),1/tmean,color='blue',label="Single")
	plt.plot(range(1,t_batch.shape[0]+1),1/tmean_batch,color='orange',label="Batch (5)")
	plt.fill_between(range(1, len(tmean)+1), 1/tmean.min(axis=0), 1/tmean.max(axis=0), color='blue', alpha=alpha)
	plt.fill_between(range(1, len(tmean)+1), 1/t_batch.min(axis=1), 1/t_batch.max(axis=1),color='orange', alpha=alpha)
	plt.xlabel("Number of robots")
	plt.ylabel("Real time factor")
	plt.legend(loc="upper left")
	plt.savefig(figurename) if figurename is not None else plt.show()

if __name__ == "__main__":
    
	# Input argument parser
	parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
	parser.add_argument('-plot', type=str, help="plot file", default=None)
	parser.add_argument('-tl', type=int, help="time limit", default=None)
	args = parser.parse_args()

	if args.plot is None:
		# Primary input parameters -- Test parameters
		n = 3 # Max number of robots
		m = 5 # Re-runs to average out
		batch = 5 # Number of batch runs
		rtfactor = 100 # Desired realtime factor (0 = #nosleep)
		tl = 200 # Length of simulation

		# Folder where data is saved
		folder = "data/time/"
		directory = os.path.dirname(folder)
		if not os.path.exists(directory): os.makedirs(directory)
		filename = (folder+"time_n%i_m%i_b%i_rt%i_tl%i" % (n,m,batch,rtfactor,tl))
		
		# Secondary input parameters -- Swarmulator configuration
		sim.runtime_setting("simulation_updatefreq", str("20"))
		sim.runtime_setting("environment", "square20")
		sim.runtime_setting("animation_updatefreq", str("25"))
		sim.runtime_setting("simulation_realtimefactor", str(rtfactor))
		sim.runtime_setting("time_limit", str(tl))

		# Run
		t, t_batch = run(n,m,batch,filename)

		# Save
		np.savez(filename, t=t, t_batch=t_batch)

	else:
		folder = "figures/time/"
		directory = os.path.dirname(folder)
		if not os.path.exists(directory): os.makedirs(directory)

		plot_evaluationtime(args.plot,figurename=folder+"evaluation_time.pdf")
		plot_realtimefactor(args.plot,args.tl,figurename=folder+"runtime_comparison.pdf")