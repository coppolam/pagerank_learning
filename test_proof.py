import numpy as np
import argparse
import test_learnmodel as l
from tools import matrixOperations as matop
from classes import pagerank_optimization as propt
import parameters
from classes import simulator, evolution, desired_states_extractor
import matplotlib
import matplotlib.pyplot as plt

# Args
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('controller', type=str, help="Controller to use")
parser.add_argument('training_folder', type=str, help="Training folder to use")
args = parser.parse_args()

# Load parameters
fitness, controller, agent, pr_states, pr_actions = parameters.get(args.controller)

# Load simulator model
sim = simulator.simulator()
sim = l.learn_model(sim, args.training_folder, discount=1.0)

# Get desired states
dse = desired_states_extractor.desired_states_extractor()
dse.load_model("data/%s/models.pkl"%controller)
des = dse.get_des(dim=pr_states,gens=10,popsize=100)

# # Optimize policy
policy = np.random.rand(pr_states,pr_actions)
policy = sim.optimize(policy, des)

# Load up sim variables locally
A = sim.A
E = sim.E
H = np.sum(A, axis=0)
with np.errstate(divide='ignore',invalid='ignore'):
	r = H.sum(axis=1) / E.sum(axis=1)
	r = np.nan_to_num(r) # Remove NaN Just in case
	alpha = r / (1 + r)
G = np.diag(alpha).dot(H) + np.diag(1-alpha).dot(E) # Google matrix
pr0 = matop.pagerank(G)

# Check des and policy aggregation
H1 = propt.update_H(A, policy)
G = np.diag(alpha).dot(H1) + np.diag(1-alpha).dot(E) # Google matrix
pr1 = matop.pagerank(G)

# Get original fitness and new fitness
f0 = propt.pagerankfitness(pr0,des)
f1 = propt.objective_function(policy, des, alpha, A, E)

# Diff plot of pagerank values
plt.bar(range(pr0[0].size),pr1[0]-pr0[0],alpha=0.5)
plt.show()

# Plot of absolute pagerank values
plt.bar(range(pr0[0].size),pr0[0],alpha=0.5)
plt.bar(range(pr0[0].size),pr1[0],alpha=0.5)
plt.show()

# Is there always at least one simplicial agent in the swarm? 
import networkx as nx

G1 = nx.from_numpy_matrix(H)
G2 = nx.from_numpy_matrix(E)

def condition_1(G,des):
	'''All des states can be reached via H'''
	# Check
	for node in range(len(G.nodes)):
		for d in des:
			if nx.has_path(G,node,d[0]) is False:
				return False # Counterexample found
	return True


def condition_2(G,static,active):
	'''Static states that are not desired can become active via the environment (Matrix E)'''
	
	# Check
	counterexampleflag = False
	for s in static:
		for a in active:
			if nx.has_path(G,s,a[0]) is False:
				print("Counterexample found for path %i to %i"%(s, a[0]))
				counterexampleflag = True
	if counterexampleflag: return False
	return True


print("\n------- MODEL -------")
print("H matrix:\n",H)
print("\nE matrix:\n",sim.E)
print("\nalpha vector:\n",alpha)

print("\n------- POLICY -------")
print(policy)

# Ignore unknown nodes with no information
d = list(nx.isolates(G1))
G1.remove_nodes_from(d)
des = np.delete(des,d)
a = 1 if pr_actions > 1 else 0
policy = np.delete(policy,d,axis=a)
static = np.argwhere(np.array(np.sum(policy,axis=a))<0.001)
active = np.argwhere(np.array(np.sum(policy,axis=a))>0.001)
happy = np.argwhere(np.array(des)>0.1)
static_unhappy = np.setdiff1d(static,happy)

c1 = condition_1(G1,happy)
c2 = condition_2(G2,static_unhappy,active)

print("\n------- STATS -------")
print("Original fitness = ",f0)
print("New fitness = ",f1)
print("Static states:\n",static.T[0])
print("Active states:\n",active.T[0])
print("Happy states:\n",happy.T[0])
print("Static and unhappy states:\n",static_unhappy.T)
print("Condition 1:",c1)
print("Condition 2:",c2)