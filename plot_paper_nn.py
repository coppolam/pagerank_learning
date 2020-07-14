#!/usr/bin/env python3
"""
Plot comparisons of the correlation between the trained NN and the validations sets
@author: Mario Coppola, 2020
"""
from tools import fileHandler as fh
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tools import prettyplot as pp

# Input argument parser
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('controller', type=str, help="(str) Controller", default=None)
args = parser.parse_args()

filename = []
name = []
if args.controller == "aggregation":
	filename.append("data/aggregation/validation_aggregation_1_1.pkl")
	filename.append("data/aggregation/validation_aggregation_2_1.pkl")
	name.append("Aggregation 1")
	name.append("Aggregation 2")
elif args.controller == "pfsm_exploration":
	filename.append("data/pfsm_exploration/validation_pfsm_exploration_1_1.pkl")
	filename.append("data/pfsm_exploration/validation_pfsm_exploration_2_1.pkl")
	filename.append("data/pfsm_exploration/validation_pfsm_exploration_mod_1_1.pkl")
	filename.append("data/pfsm_exploration/validation_pfsm_exploration_mod_2_1.pkl")
	name.append("Oriented 1")
	name.append("Oriented 2")
	name.append("Oriented mod 1")
	name.append("Oriented mod 2")
elif args.controller == "forage":
	filename.append("data/forage/validation_forage_1_1.pkl")
	filename.append("data/forage/validation_forage_2_1.pkl")
	name.append("Forage 1")
	name.append("Forage 2")
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
plt = pp.setup()
for i,d in enumerate(data):
	plt.plot(d[0],label=name[i])
	plt.fill_between(range(len(d[0])),
	np.array(d[0]) - np.array(d[1]), np.array(d[0]) + np.array(d[1]), alpha=0.2)

plt.xlabel("Epoch")
plt.ylabel("Correlation")
plt.legend()
plt = pp.adjust(plt)
plt.show()
