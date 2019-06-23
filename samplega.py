#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 18:40:27 2019

@author: mario
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

## Input
data_write_folder = 'data/gridmaze/evolutions/' #Where the final data will be stored
save_data = 0

## Libraries
import numpy as np
np.set_printoptions(suppress=True) #prevent numpy exponential notation on print, default False
import matplotlib as plot 
import random
import networkx as nx
from pybrain.optimization import GA

# Subclass of GA with own definitions
class GA_alt(GA):
    def copy(self):         return GA_alt(self.x)

def set_runtime_ID():
    r = random.randrange(1000)
    return r

def objF(x):
    e = (0, 2, 1, 4, 5, 4, 3, 6, 3)
    t = (2, 3, 4, 2, 1, 2, 4, 4, 6)
    elist = list(zip(e,t,x))
    G = nx.DiGraph()
    G.add_weighted_edges_from(elist)
    pr = nx.pagerank(G,weight='weight',alpha=0.1)
    pr_values = list(pr.values())
    fitness = pr_values[1]/np.mean(pr_values)
    return fitness
#   return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2 # (Test) Himmelblau's function

def extract_history(l):
    # Extract fitness history
    fitness_history = []
    for x in range(0, l.numLearningSteps):
       fitness_history.append(max(l._allGenerations[x][1]))
    return fitness_history

def initialize_evolution_parameters(l):
    l.verbose = False # Verbose, defined on top
    l.maximize = True # Maximize the fitness function
    l.storeAllPopulations = True # Keep history
    l.populationSize = 20 # Population
    l.maxLearningSteps = 1000 # Generations
    return l

runtime_ID = set_runtime_ID()
print("Starting Optimization \nRuntime ID =", runtime_ID)

## Learning parameters
x0 = np.ones(8) # Initialize to ones
lim = list(zip(list(np.zeros(8)),list(np.ones(8)))) # Bind values
GA_alt.xBound = lim # Set limits
GA_alt.elitism = True # Use elite mem
l = GA_alt(objF, x0) # Set up GA (alternative subclass)
l = initialize_evolution_parameters(l)

## Learn
l.learn()

## Evaluate output
fitness_history = extract_history(l)
print(l.bestEvaluable)
plot.pyplot.plot(fitness_history)

#np.save('test')