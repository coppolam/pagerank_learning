#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 15:03:19 2019

@author: mario
"""
import numpy as np
import auxiliary as aux
import networkx as nx

def pagerank(G, tol=1e-6):
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

# TODO: Make
def alpha():
    # something something
    return alpha

# TODO: Make
def GSactive(Q,states):
    nstates = 
    for x in range(0,nstates):
        
        for j in range(1,bw):
        
        
    # something something
    return Ga

# TODO: Make
def GSpassive(states):
    # something something
    return Gp

def GoogleMatrix(alpha, H, E):
    alpha_mat = alpha   #    aM = np.asmatrix(np.diag(alpha,8)) 
    H = nx.adjacency_matrix(Ga) # make sure this has weights
    E = nx.adjacency_matrix(Gp) # make sure this has weights
    Em = aux.normalize_rows(E) # Normalize environment matrix E
    
    # Get D for columns where H = 0
    z = np.where(x <= 0.00001)[0] # Find transitions with no action
    D = np.matlib.zeros((M,M)) # Blank slate for matrix D 
    D[z,:] = E[z,:] # Account for blocked states in matrix D
    
    # Normalize H
    S = aux.normalize_rows(Hm + D) # Normalize policy matrix S = H + D
    
    G = np.matmul(aM, S) + np.matmul(np.subtract(np.eye(aM.shape[0]), aM), Em)
    return G

def objF(x):
    Ga = GSactive(Q,states)
    Gp = GSpassive(x)
    # TODO: proper method
    
    pr = pagerank(GoogleMatrix(alpha, H, E)) # Evaluate Pagerank vector 
    fitness = fitness_function(pr) # Get fitness
#    print(fitness) # DEBUG
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
    l.storeAllPopulations = True # Keep history
    l.populationSize = 10 # Population
    l.maxLearningSteps = 100 # Generations
    return l
