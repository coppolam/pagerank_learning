from tools import fileHandler as fh
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

mode = 2

filename = []
name = []
if mode == 1:
	filename.append("data/aggregation/validation_aggregation_1_1.pkl")
	filename.append("data/aggregation/validation_aggregation_2_1.pkl")
	name.append("aggregation 1")
	name.append("aggregation 2")
elif mode == 2:
	filename.append("data/pfsm_exploration/validation_pfsm_exploration_1_1.pkl")
	filename.append("data/pfsm_exploration/validation_pfsm_exploration_2_1.pkl")
	filename.append("data/pfsm_exploration/validation_pfsm_exploration_mod_1_1.pkl")
	filename.append("data/pfsm_exploration/validation_pfsm_exploration_mod_2_1.pkl")
	name.append("pfsm exploration 1")
	name.append("pfsm exploration 2")
	name.append("pfsm exploration mod 1")
	name.append("pfsm exploration mod 2")
elif mode == 3:
	filename.append("data/aggregation/validation_forage_1_1.pkl")
	filename.append("data/aggregation/validation_forage_2_1.pkl")
	name.append("forage 1")
	name.append("forage 2")
else:
	print("No valid mode!!!!")

def process(file):
	a = []
	for e in file: a.append(np.nanmean(e))
	return a

# Load files and data
file, data = [], []
for f in filename:file.append(fh.load_pkl(f))
for f in file: data.append(process(f))

# Plot
for i,d in enumerate(data): plt.plot(d,label=name[i])
plt.xlabel("Epoch")
plt.ylabel("Correlation")
plt.legend()
plt.gcf().subplots_adjust(bottom=0.15)
plt.show()
