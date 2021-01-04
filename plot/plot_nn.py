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
	parser = argparse.ArgumentParser(
			description='Simulate a task to gather \
						 the data for optimization')
	parser.add_argument('controller', type=str, default=None,
			help="(str) Controller")
	parser.add_argument('-format', type=str, default="pdf", 
			help="(str) Figure file format, default pdf")
	args = parser.parse_args(args)

	# Default character init
	df = "data/%s/validation_%s_"%(args.controller,args.controller)

	filenames = []
	name = []
	if args.controller == "aggregation":
		filenames.append(df+"aggregation_1.pkl")
		filenames.append(df+"aggregation_2.pkl")
		filenames.append(df+"aggregation_3.pkl")
		# Plot names
		name.append("VS 1")
		name.append("VS 2")
		name.append("VS 3")
	elif args.controller == "pfsm_exploration":
		# Files where it evaluates against itself
		filenames.append(df+"pfsm_exploration_1.pkl")
		filenames.append(df+"pfsm_exploration_2.pkl")
		filenames.append(df+"pfsm_exploration_3.pkl")
		# Files where it evaluates against different dynamics
		filenames.append(df+"pfsm_exploration_mod_1.pkl")
		filenames.append(df+"pfsm_exploration_mod_2.pkl")
		filenames.append(df+"pfsm_exploration_mod_3.pkl")
		name.append("VS 1 (B1)")
		name.append("VS 2 (B1)")
		name.append("VS 3 (B1)")
		name.append("VS 1 (B2)")
		name.append("VS 2 (B2)")
		name.append("VS 3 (B2)")
	elif args.controller == "pfsm_exploration_mod":
		# Files where it evaluates against itself
		filenames.append(df+"pfsm_exploration_mod_1.pkl")
		filenames.append(df+"pfsm_exploration_mod_2.pkl")
		filenames.append(df+"pfsm_exploration_mod_3.pkl")
		# Files where it evaluates against different dynamics
		filenames.append(df+"pfsm_exploration_1.pkl")
		filenames.append(df+"pfsm_exploration_2.pkl")
		filenames.append(df+"pfsm_exploration_3.pkl")
		name.append("VS 1 (B2)")
		name.append("VS 2 (B2)")
		name.append("VS 3 (B2)")
		name.append("VS 1 (B1)")
		name.append("VS 2 (B1)")
		name.append("VS 3 (B1)")
	elif args.controller == "forage":
		filenames.append(df+"forage_1.pkl")
		filenames.append(df+"forage_2.pkl")
		filenames.append(df+"forage_3.pkl")
		name.append("VS 1")
		name.append("VS 2")
		name.append("VS 3")
	else:
		print("Not a valid mode!!!!")

	def process(file):
		m = [] # Mean
		s = [] # Standard deviation
		for e in file:
			m.append(np.nanmean(e))
			s.append(np.nanstd(e))
		return m, s

	# Load and process the data in each file
	data = []
	for f in filenames:
		data.append(process(fh.load_pkl(f)))

	# Plot
	color = ["blue", "red", "green", "black", "magenta", "lime"]
	plt = pp.setup()
	for i,d in enumerate(data):

		# Plot line
		plt.plot(d[0],label=name[i],color=color[i])

		# Error margin
		plt.fill_between(range(len(d[0])),
			np.array(d[0])-np.array(d[1]),
			np.array(d[0])+np.array(d[1]),
			alpha=0.2,
			color=color[i])
	
	# Plot deets
	plt.xlabel("Simulation")
	plt.ylabel("Correlation")
	plt.legend()
	plt = pp.adjust(plt)

	# Save
	fname = "nn_correlation_%s.%s"%(args.controller,args.format)
	folder = "figures/nn/"
	if not os.path.exists(os.path.dirname(folder)):
		os.makedirs(os.path.dirname(folder))
	filename_raw = os.path.splitext(os.path.basename(fname))[0]
	plt.savefig(folder+"%s.%s"%(filename_raw,args.format))
	plt.close()

if __name__ == "__main__":
	main(sys.argv[1:])