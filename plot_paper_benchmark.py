from tools import fileHandler as fh
import argparse, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tools import prettyplot as pp

def plot_benchmark(f):
	for i in f: plt.hist(i, alpha=0.1)

def plot_new(f):
	plt.hist(f, alpha=0.5)

def benchmark(benchmarkfile,new=None,filename=None):
	plt = pp.setup()
	f = fh.load_pkl(benchmarkfile)
	fn = fh.load_pkl(new)
	plot_benchmark(f)
	if new is not None: plot_new(fn)
	plt.xlabel("Fitness [-]")
	plt.ylabel("Frequency")
	plt = pp.adjust(plt)
	# Save or show
	if filename is not None:
		folder = os.path.dirname(benchmarkfile) + "/figures/"
		if not os.path.exists(os.path.dirname(folder)): os.makedirs(os.path.dirname(folder))
		filename_raw = os.path.splitext(os.path.basename(file))[0]
		plt.savefig(folder+"%s.pdf"%filename_raw)
	else: plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
	parser.add_argument('benchmarkfile', type=str, help="(str) Benchmark fitness logs")
	parser.add_argument('optimizedfile', type=str, help="(str) Optimized fitness logs")
	args = parser.parse_args()
	benchmark(args.benchmarkfile,new=args.optimizedfile)