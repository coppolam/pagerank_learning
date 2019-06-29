#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

"""
Script to run a PageRank based optimization

Created on Wed Jun 19 18:40:27 2019
@author: Mario Coppola
"""

## Libraries
import numpy as np
np.set_printoptions(suppress=True) #prevent numpy exponential notation on print, default False
import matplotlib as plot
from pybrain.optimization import GA

graph = True
import_mat_data = True
M = 8

if import_mat_data:
    alpha = np.loadtxt('inputs/alpha.in')
    H = np.asmatrix(np.reshape(np.loadtxt('inputs/H.in'),(M,M)))
    E = np.asmatrix(np.reshape(np.loadtxt('inputs/E.in'),(M,M)))
else:
    # Test
    alpha = np.matlib.eye(M,M) * 0.5
    H = np.matlib.ones((M,M)) * 0.2
    E = np.matlib.ones((M,M)) * 0.2

def normalize_rows(mat):
    row_sums = np.squeeze(np.asarray(mat.sum(axis=1)))
    mat_norm = mat / row_sums[:, np.newaxis]
#    if not np.all(row_sums):
#        print(mat)
#        print(row_sums)
    return mat_norm

def pagerank(G, tol=1e-8):
    # Iterative procedure
    n = G.shape[0] # Size of G
    pr = 1 / n * np.ones((1, n)) # Initialize Pagerank vector
    residual = 1 # Residual (initialize high, doesn't really matter)
    while residual >= tol:
        pr_previous = pr
        pr = np.matmul(pr,G) # Pagerank formula
        residual = np.linalg.norm(np.subtract(pr,pr_previous))
    return np.squeeze(np.asarray(pr))

def fitness_function(pr):
    f = np.mean(pr[2:])/np.mean(pr)
    return f

def objF(x):
    aM = np.asmatrix(np.diag(alpha)) # Alpha vector in matrix format
    Hm = np.asmatrix(np.asarray(H) * x[:, np.newaxis]) # Multiply probability of transition with probability of policy
    z = np.where(x <= 0.001)[0] # Find transitions with no action
    D = np.matlib.zeros((M,M)) # Blank slate for matrix D 
    D[z,:] = E[z,:] # Account for blocked states in matrix D
    S = normalize_rows(Hm + D) # Normalize policy matrix S = H + D
    Em = normalize_rows(E) # Normalize environment matrix E
    G = np.matmul(aM, S) + np.matmul(np.subtract(np.eye(aM.shape[0]), aM), Em)
    pr = pagerank(G) # Evaluate pagerank vector 
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

## Learning parameters
x0 = np.ones(M)/2 # Initialize to ones
GA.xBound = list(zip(list(np.zeros(M)),list(np.ones(M)))) # Set limits
GA.elitism = True # Use elite mem
l = GA(objF, x0) # Set up GA (alternative subclass)
l = initialize_evolution_parameters(l)

## Learn
l.learn()

## Evaluate output
if graph:
    print(l.bestEvaluable)
    fitness_history = extract_history(l)
    plot.pyplot.plot(fitness_history)

## Store
np.savetxt('outputs/weights.out', l.bestEvaluable, delimiter=',', fmt='%2.3f')
