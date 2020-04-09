import numpy as np
from tools import fileHandler as fh
import matplotlib.pyplot as plt

data = np.load("data/2020_04_09_17:50:04_validation.npz")

fitness = data['arr_0'].astype(float)
fitness_n = data['arr_1']
fitness_n = np.delete(fitness_n,82)
fitness_n = fitness_n.astype(float)

# a = np.histogram(list(fitness))
_ = plt.hist(fitness, alpha=0.5, density=True, stacked=True)  # arguments are passed to np.histogram
_ = plt.hist(fitness_n, alpha=0.5, density=True, label='new')  # arguments are passed to np.histogram
plt.title("Results")
plt.legend(loc='upper right')
plt.show()

data = np.load("data/aggregation/2020_04_08_17:19:55_validation.npz")

fitness = data['arr_0'].astype(float)
fitness_n = data['arr_1']
fitness_n = fitness_n.astype(float)

# a = np.histogram(list(fitness))
_ = plt.hist(fitness, alpha=0.5, density=True, stacked=True)  # arguments are passed to np.histogram
_ = plt.hist(fitness_n, alpha=0.5, density=True, label='new')  # arguments are passed to np.histogram
plt.title("Results")
plt.legend(loc='upper right')
plt.show()
