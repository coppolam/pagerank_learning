#!/usr/bin/env python3
"""
Plot the benchmark results against the optimized ones
@author: Mario Coppola, 2020
"""

import argparse, os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tools import fileHandler as fh
from tools import prettyplot as pp

def plot_benchmark(f): 
	data = []
	for i,d in enumerate(f):
		data = np.append(data,d)
		# plt.hist(d,alpha=0.1, label='Random' if i==0 else None)
	# plt.boxplot(data)
	return data

# def plot_new(f,l=None):
	# plt.boxplot(f)
# 	# plt.hist(f, alpha=0.9, label=l)

def benchmark(benchmarkfile,om=None,om_2=None,em=None,filename=None,fileformat="pdf"):
	plt = pp.setup()
	f = fh.load_pkl(benchmarkfile)
	bm = plot_benchmark(f)
	if om_2 is None: data = [bm, fh.load_pkl(om),fh.load_pkl(em)]
	else: data = [bm, fh.load_pkl(om),fh.load_pkl(om_2),fh.load_pkl(em)]
	plt.boxplot(data)
	plt.ylabel("Fitness $F_g$")
	labels = ["Random","PR","Evolved"]
	ax = plt.gca()
	ax.set_xticklabels(labels)
	plt = pp.adjust(plt)

	# Save or show
	if filename is not None:
		folder = "figures/benchmark/"
		if not os.path.exists(os.path.dirname(folder)):
			os.makedirs(os.path.dirname(folder))
		filename_raw = os.path.splitext(os.path.basename(filename))[0]
		plt.savefig(folder+"%s.%s"%(filename_raw,fileformat))
		plt.close()
	else:
		plt.show()
		plt.close()

def main(args):
	parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
	parser.add_argument('controller', type=str, help="(str) Benchmark fitness logs")
	parser.add_argument('-format', type=str, help="(str) Controller", default="pdf")
	args = parser.parse_args(args)	
	
	filename = []
	name = []
	om_2 = None
	pref = "data/" + args.controller + "/benchmark_"
	if args.controller == "aggregation":
		bm = pref + "random_aggregation_t200_r30_runs100_1.pkl"
		om = pref + "optimized_aggregation_t200_r30_runs100_1.pkl"
		em = pref + "evolution_aggregation_t200_r30_runs100_1.pkl"
	elif args.controller == "pfsm_exploration":
		bm = pref + "random_pfsm_exploration_t200_r30_runs100_1.pkl"
		om = pref + "optimized_pfsm_exploration_t200_r30_runs100_1.pkl"
		em = pref + "evolution_pfsm_exploration_t200_r30_runs100_1.pkl"
	elif args.controller == "pfsm_exploration_mod":
		bm = pref + "random_pfsm_exploration_mod_t200_r30_runs100_1.pkl"
		om = pref + "optimized_pfsm_exploration_mod_t200_r30_runs100_1.pkl"
		em = pref + "evolution_pfsm_exploration_mod_t200_r30_runs100_1.pkl"
	elif args.controller == "forage":
		bm = pref + "random_forage_t500_r20_runs100_1.pkl"
		om = pref + "optimized_forage_t500_r20_runs100_1.pkl"
		em = pref + "evolution_forage_t500_r20_runs100_1.pkl"
	else:
		print("\n\n\nNot a valid\n\n\n")

	f = "benchmark_%s.%s"%(args.controller,args.format)
	benchmark(bm, om=om, om_2=om_2, em=em, filename=f)

if __name__ == "__main__":
	main(sys.args[1:])
	