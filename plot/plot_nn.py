#!/usr/bin/env python3
"""
Plot comparisons of the correlation between the trained NN and the validations sets
@author: Mario Coppola, 2020
"""
from tools import fileHandler as fh
import argparse, os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tools import prettyplot as pp

def main(args):
	# Input argument parser
	parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
	parser.add_argument('controller', type=str, help="(str) Controller", default=None)
	parser.add_argument('-format', type=str, help="(str) Controller", default="pdf")
	args = parser.parse_args(args)

	# Default character init
	df = "data/%s/validation_%s_"%(args.controller,args.controller)

	filename = []
	name = []
	if args.controller == "aggregation":
		filename.append(df+"aggregation_1.pkl")
		filename.append(df+"aggregation_2.pkl")
		# Plot names
		name.append("VS 1")
		name.append("VS 2")
	elif args.controller == "pfsm_exploration":
		# Files where it evaluates against itself
		filename.append(df+"pfsm_exploration_1.pkl")
		filename.append(df+"pfsm_exploration_2.pkl")
		# Files where it evaluates against different dynamics
		filename.append(df+"pfsm_exploration_mod_1.pkl")
		filename.append(df+"pfsm_exploration_mod_2.pkl")
		name.append("VS 1, Study Case C1")
		name.append("VS 2, Study Case C1")
		name.append("VS 1, Study Case C2")
		name.append("VS 2, Study Case C2")
	elif args.controller == "pfsm_exploration_mod":
		# Files where it evaluates against itself
		filename.append(df+"pfsm_exploration_mod_1.pkl")
		filename.append(df+"pfsm_exploration_mod_2.pkl")
		# Files where it evaluates against different dynamics
		filename.append(df+"pfsm_exploration_1.pkl")
		filename.append(df+"pfsm_exploration_2.pkl")
		name.append("VS 1, Study Case C2")
		name.append("VS 2, Study Case C2")
		name.append("VS 1, Study Case C1")
		name.append("VS 2, Study Case C1")
	elif args.controller == "forage":
		filename.append(df+"forage_1.pkl")
		filename.append(df+"forage_2.pkl")
		name.append("VS 1")
		name.append("VS 2")
	else:
		print("Not a valid mode!!!!")

	def process(file):
		m = []
		s = []
		for e in file:
			m.append(np.nanmean(e))
			s.append(np.nanstd(e))
		return m, s

	# Load files and data
	file, data = [], []
	for f in filename:file.append(fh.load_pkl(f))
	for f in file: data.append(process(f))

	# Plot
	color = ["blue","red","green","black"]
	plt = pp.setup()
	for i,d in enumerate(data):
		plt.plot(d[0],label=name[i],color=color[i]) # Plot line
		plt.fill_between(range(len(d[0])),
			np.array(d[0])-np.array(d[1]),np.array(d[0])+np.array(d[1]),
			alpha=0.2, color=color[i]) # Error margin

	plt.xlabel("Simulation")
	plt.ylabel("Correlation")
	plt.legend()
	plt = pp.adjust(plt)

	# Save or show
	fname = "nn_correlation_%s.%s"%(args.controller,args.format)
	if fname is not None:
		folder = "figures/nn/"
		if not os.path.exists(os.path.dirname(folder)): os.makedirs(os.path.dirname(folder))
		filename_raw = os.path.splitext(os.path.basename(fname))[0]
		plt.savefig(folder+"%s.%s"%(filename_raw,args.format))
		plt.close()
	else:
		plt.show()
		plt.close()

if __name__ == "__main__":
	main(sys.argv[1:])