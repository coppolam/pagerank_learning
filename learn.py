#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:07:37 2019

@author: mario
"""
import numpy as np
np.set_printoptions(suppress=True) #prevent numpy exponential notation on print, default False

def normalize_rows(mat):
    row_sums = np.squeeze(np.asarray(mat.sum(axis=1)))
    mat_norm = mat / row_sums[:, np.newaxis]
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