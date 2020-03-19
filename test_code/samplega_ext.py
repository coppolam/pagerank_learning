#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

"""
Script to run a PageRank based optimization

Created on Wed Jun 19 18:40:27 2019
@author: Mario Coppola
"""

# from IPython import get_ipython
# get_ipython().magic('reset -sf')

## Libraries
import numpy as np
np.set_printoptions(suppress=True) #prevent numpy exponential notation on print, default False
import matplotlib.pyplot as plt
import scipy as sp
import random
import networkx as nx
from pybrain.optimization import GA
import graph as gt

G_active = nx.complete_graph(8)
G_passive = nx.complete_graph(8)
#

H = nx.adjacency_matrix(G_active)
E = nx.adjacency_matrix(G_passive)
H = H.astype(np.double)
E = E.astype(np.double)

## Input
# data_write_folder = 'data/gridmaze/evolutions/' #Where the final data will be stored
# save_data = 0

# def set_runtime_ID():
#     r = random.randrange(100)
#     return r

# Subclass of GA with own definitions
#class GA_alt(GA):
#    def copy(self):         return GA_alt(self.x)

def pagerank(x):
    return x

def objF(x):
    e = (0, 2, 1, 4, 5, 4, 3, 6, 3)
    t = (2, 3, 5, 2, 1, 2, 4, 1, 6)
    G = gt.make_graph(e,t,x)
    
    pr = nx.pagerank(G,weight='weight',alpha=0.1)
    pr_values = list(pr.values())
    fitness = pr_values[1]/np.mean(pr_values)
    return fitness

def extract_history(l):
    # Extract fitness history
    fitness_history = []
    for x in range(0, l.numLearningSteps):
       fitness_history.append(max(l._allGenerations[x][1]))
    return fitness_history

def initialize_evolution_parameters(l):
    l.verbose = False # Verbose, defined on top
    l.maximize = True # Maximize the fitness function
    # l.storeAllPopulations = True # Keep history
    l.populationSize = 10 # Population
    l.maxLearningSteps = 100 # Generations
    return l

# runtime_ID = set_runtime_ID()
# print("Starting Optimization \nRuntime ID =", runtime_ID)

## Learning parameters
x0 = np.ones(8) # Initialize to ones
lim = list(zip(list(np.zeros(8)),list(np.ones(8)))) # Bind values
GA.xBound = lim # Set limits
GA.elitism = True # Use elite mem
l = GA(objF, x0) # Set up GA (alternative subclass)
l = initialize_evolution_parameters(l)

## Learn
l.learn()

## Evaluate output
# fitness_history = extract_history(l)
print(l.bestEvaluable)

# Plot (if in Anaconda)
plt.plot([1,2,3])
# np.save('test')
plt.show()