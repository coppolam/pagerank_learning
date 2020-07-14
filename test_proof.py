import numpy as np
import argparse
from classes import simulator
from classes import pagerank_optimization as pr

# Args
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('file', type=str, help="Simulation file to use")
args = parser.parse_args()

# Import simulator API 
sim = simulator.simulator()
sim.load(args.file)


# Calculate predicted H1
# if args.controller == "aggregation":
#     fitness = "aggregation_clusters"
#     pol0 = np.ones((8,1))/2 # all = 1/2
# elif args.controller == "pfsm_exploration":
#     fitness = "aggregation_clusters"
#     pol0 = np.ones((16,8))/8 # all = 1/8
# elif args.controller == "pfsm_exploration_mod":
#     fitness = "aggregation_clusters"
#     pol0 = np.ones((16,8))/8 # all 1/8
# elif args.controller == "forage":
#     fitness = "food"
#     pol0 = np.ones((16,1))/2 # all = 1/2
# else:    
#     ValueError("Unknown controller!")
des = np.ones((1,16))
pol0 = np.ones((16,8))/8 # all = 1/8
pol  = sim.optimize(pol0,des)

b0 = pr.update_b(sim.A,pol0)
b1 = pr.update_b(sim.A,pol)

H1 = np.divide(b1, b0, where=b0!=0) * sim.H

import networkx as nx
from tools import graph

G1 = nx.DiGraph(H1)
G2 = nx.Digraph(sim.E)

# Can agents always become active and go to any state on their own


# Is there always at least one simplicial agent in the swarm? 


# Can the desired states coexist
# e.g. aggregation
# desired states with no neighbors cannot coexist with
# desired state with neighbors
