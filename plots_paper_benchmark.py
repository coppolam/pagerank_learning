from tools import fileHandler as fh
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import os

def plot_benchmark(f):
	alpha = 0.1
	for i in f: plt.hist(i, alpha=alpha)

def plot_new(f):
	alpha=1.0
	plt.hist(f, alpha=alpha)

def benchmark(benchmarkfile,new=None):
	plt.figure(figsize=(6,3))
	f = fh.load_pkl(benchmarkfile)
	plot_benchmark(f)
	if new is not None: plot_new(new)
	plt.xlabel("Fitness [-]")
	plt.ylabel("Frequency")
	plt.gcf().subplots_adjust(bottom=0.15)
	plt.show()
	# folder = os.path.dirname(file) + "/figures/"
	# if not os.path.exists(os.path.dirname(folder)): os.makedirs(os.path.dirname(folder))
	# filename_raw = os.path.splitext(os.path.basename(file))[0]
	# plt.savefig(folder+"%s.pdf"%filename_raw)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
	parser.add_argument('file', type=str, help="(str) Training data folder")
	args = parser.parse_args()
	benchmark(args.file)