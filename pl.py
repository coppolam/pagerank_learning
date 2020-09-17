# Code to plot the policy

import numpy as np
from tools import fileHandler as fh
np.set_printoptions(suppress=True)

d = np.load("data/pfsm_exploration/optimization.npz")

np.set_printoptions(precision=2)
print(d["policy"])

import matplotlib
from matplotlib import pyplot as plt

plt.imshow(d["policy"])
plt.colorbar()
plt.show()