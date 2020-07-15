import numpy as np
import argparse
from classes import simulator
import test_learnmodel as l
from tools import matrixOperations as matop
from classes import pagerank_optimization as propt

# Args
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('file', type=str, help="Simulation file to use")
args = parser.parse_args()

# Load simulator model
sim = simulator.simulator()
sim = l.learn_model(sim, args.file, discount=1.0)

A = sim.A
E = sim.E
H = np.sum(A, axis=0)

with np.errstate(divide='ignore',invalid='ignore'):
	r = H.sum(axis=1) / E.sum(axis=1)
	r = np.nan_to_num(r) # Remove NaN Just in case
	alpha = r / (1 + r)

G = np.diag(alpha).dot(H) + np.diag(1-alpha).dot(E) # Google matrix

print(H)
print(sim.E)
pr0 = matop.pagerank(G)

# Can agents always become active and go to any state on their own

# debug des and policy aggregation
des = np.array([0,1,1,1,1,1,1,1])
policy = np.array([1,0,0,0,0,0,0,0])

H1 = propt.update_H(A, policy)
G = np.diag(alpha).dot(H1) + np.diag(1-alpha).dot(E) # Google matrix
pr1 = matop.pagerank(G)

f0 = propt.pagerankfitness(pr0,des)
f1 = propt.pagerankfitness(pr1,des)

print(f0,pr0)
print(f1,pr1)

import matplotlib
import matplotlib.pyplot as plt
plt.bar(range(pr0[0].size),pr0[0],alpha=0.5)
plt.bar(range(pr1[0].size),pr1[0],alpha=0.5)
plt.show()


# Is there always at least one simplicial agent in the swarm? 

# Can the desired states coexist
# e.g. aggregation
# desired states with no neighbors cannot coexist with
# desired state with neighbors
