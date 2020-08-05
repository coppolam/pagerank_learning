#!/usr/bin/env python3
"""
Plot the benchmark results against the optimized ones
@author: Mario Coppola, 2020
"""

import argparse, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tools import fileHandler as fh
from tools import prettyplot as pp

def plot_benchmark(f): 
	for i,d in enumerate(f):
		plt.hist(d,alpha=0.1, label='Random policies' if i==0 else None)

def plot_new(f):
	plt.hist(f, alpha=0.9, label='Optimized policy')

def benchmark(benchmarkfile,new=None,filename=None,fileformat="pdf"):
	plt = pp.setup()
	f = fh.load_pkl(benchmarkfile)
	fn = fh.load_pkl(new)
	plot_benchmark(f)
	if new is not None: plot_new(fn)
	plt.xlabel("Fitness $F_g$")
	plt.ylabel("Occurrences")
	plt = pp.adjust(plt)
	plt.legend()
	name.append("Random policies")
	name.append("Optimized ")

	# Save or show
	if filename is not None:
		folder = "figures/benchmark/"
		if not os.path.exists(os.path.dirname(folder)):
			os.makedirs(os.path.dirname(folder))
		filename_raw = os.path.splitext(os.path.basename(filename))[0]
		plt.savefig(folder+"%s.%s"%(filename_raw,fileformat))
	else:
		plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
	parser.add_argument('controller', type=str, help="(str) Benchmark fitness logs")
	parser.add_argument('-format', type=str, help="(str) Controller", default="pdf")
	args = parser.parse_args()	
	
	filename = []
	name = []
	if args.controller == "aggregation":
		bm = "data/aggregation/benchmark_random_aggregation_t200_r30_runs100_1.pkl"
		om = "data/aggregation/benchmark_optimized_aggregation_t200_r30_runs100_1.pkl"
	elif args.controller == "pfsm_exploration":
		bm = "data/pfsm_exploration/benchmark_random_pfsm_exploration_t200_r30_runs100_1.pkl"
		om = "data/pfsm_exploration/benchmark_optimized_pfsm_exploration_t200_r30_runs100_1.pkl"
	elif args.controller == "pfsm_exploration_mod":
		bm = "data/pfsm_exploration_mod/benchmark_random_pfsm_exploration_mod_t200_r30_runs100_1.pkl"
		om = "data/pfsm_exploration_mod/benchmark_optimized_pfsm_exploration_mod_t200_r30_runs100_1.pkl"
	elif args.controller == "forage":
		bm = "data/forage/benchmark_random_forage_t500_r20_runs100_1.pkl"
		om = "data/forage/benchmark_optimized_forage_t500_r20_runs100_1.pkl"
	else:
		print("\n\n\nNot a valid\n\n\n")

	f = "benchmark_%s.%s"%(args.controller,args.format)
	benchmark(bm,new=om,filename=f)