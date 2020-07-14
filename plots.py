from tools import fileHandler as fh
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Parser
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('validation_file', type=str, help="(str) Save folder")
args = parser.parse_args()

# Load data
v = fh.load_pkl(args.validation_file)

# Compress mean correlation over dataset
a = []
for e in v: a.append(np.nanmean(e))

print(a)

# Plot
plt.plot(a)
plt.show()