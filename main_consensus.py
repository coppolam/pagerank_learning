#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 18:40:27 2019

@author: Mario Coppola
"""

import graph
import numpy as np
np.set_printoptions(suppress=True) #prevent numpy exponential notation on print, default False
import auxiliary as aux
import graph as gt

# Parameters of the evolution
evo = {
    'generations_max': 1000,
    'mutation_rate': 0.1, # % Mutation rate
    'elite': 0.3, # % Percentage of population that is elite
    'parents': 0.4, # % Percentage of population that reproduces
    'mutate': 0.3, # % Percentage of population that mutates
    'population_size': 10  # Population size in the evolutionary task
}

# Parameters of the consensus task
task = {
    'max_neighbors': 8, # Maximum neighbors that the agent can see
    'm': 2 # Number of choices
}

folder = '../pagerank_learning/'

# Define the fitness function to optimize for
def fitness_function(pr):
    f = np.mean(pr[2:])/np.mean(pr)
    return f

def objF(x):
    aM = np.asmatrix(np.diag(alpha)) # Alpha vector in matrix format
    Hm = np.asmatrix(np.asarray(H) * x[:, np.newaxis]) # Multiply probability of transition with probability of policy
    z = np.where(x <= 0.00001)[0] # Find transitions with no action
    D = np.matlib.zeros((M,M)) # Blank slate for matrix D
    D[z,:] = E[z,:] # Account for blocked states in matrix D
    S = aux.normalize_rows(Hm + D) # Normalize policy matrix S = H + D
    Em = aux.normalize_rows(E) # Normalize environment matrix E
    G = np.matmul(aM, S) + np.matmul(np.subtract(np.eye(aM.shape[0]), aM), Em)
    pr = gt.pagerank(G) # Evaluate pagerank vector
    fitness = fitness_function(pr) # Get fitness
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
    if graph:
        l.storeAllPopulations = True # Keep history
    l.populationSize = 10 # Population
    l.maxLearningSteps = 100 # Generations
    return l

print "----- Starting optimization -----"
runtime_ID = aux.set_runtime_ID()
print "Runtime ID:",runtime_ID
