#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 18:40:27 2019

@author: Mario Coppola
"""

# Standard libraries
import numpy as np
import itertools as tools
import scipy as sp
from pybrain.optimization import GA
import matplotlib.pyplot as plot

# Own libraries
import auxiliary as aux
import graph as gt
import networkx as nx

# Top level settings
np.set_printoptions(suppress=True) # Prevent numpy exponential notation on print, default False

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

# Make a vector of states
def make_states(n_choices,max_neighbors):
    # Construct neighbor pairings list
    ext_states = np.array(list(tools.product(range(0,max_neighbors+1),repeat=n_choices))) # Make list of neighbor combinations
    ext_states = ext_states[np.where(sp.sum(ext_states,1)<=max_neighbors)[0]] # Remove pairs with more than max_neighbors neighbors
    ext_states = ext_states[np.where(sp.sum(ext_states,1)>0)[0]] # Remove pairs with 0 neighbors

    # Construct list of own states
    int_states = np.array(np.repeat(np.arange(n_choices),np.size(ext_states,0)))

    # Combine into a single array of states
    ext_states = np.tile(ext_states, (n_choices, 1)) # Tile the external states
    states = np.append(int_states[:, np.newaxis],ext_states,axis=1) # Concatenate into final state vector
    neighbors = sp.sum(ext_states,1) # Extract vector with number of neighbors

    return states, neighbors

# Find the idx of the desired states
def find_desired_states_idx(states):
    desired_states_idx = []
    for x in np.arange(1,np.size(states,1)):
        neighbors_in_agreement_idx = np.where(states[:, np.setdiff1d(np.arange(1, np.size(states, 1)), x)] == 0)[0] # States with all neighbors in agreement for choice x
        current_opinion_idx = np.where(states[:, 0] == x-1 ) # States where you are of choice x
        desired_states_idx.extend(np.intersect1d(neighbors_in_agreement_idx,current_opinion_idx))
    desired_states_idx
    return desired_states_idx

def ismember(A, B):
    return [ np.sum(a == B) for a in A ]

def init_policy(states,desired_states_idx):
    P0 = np.full((np.size(states, 0), np.size(states, 1) - 1), 1.0 / (np.size(states, 1) - 1))
    P0[desired_states_idx] = 0
    # P0 = np.delete(P0,desired_states_idx,axis=0) # Remove pairs with more than max_neighbors neighbors
    return P0

def GS_active(Q, states):
    n_states = np.size(states, 0)
    n_choices = np.size(Q, 1)
    s = [] # Starting nodes
    t = [] # End nodes
    w = [] # Weights
    for i in range(0, n_states): # For each state, we will extract the possible edges and weights
        s_i = states[i, :] # Selected state
        indexes = [] # Indexes
        for j in range(0, n_choices):
            indexes.extend(np.where((states == np.append(j,s_i[range(1,n_choices+1)])).all(axis=1))[0])
        s.extend(int(i) * np.ones((n_choices,),dtype=int))
        t.extend(indexes)
        w.extend(Q[i, :])

    Ga = gt.make_digraph(s, t, w) # Make the digraph
    return Ga, s

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

def initialize_evolution_parameters(l,evo):
    l.verbose = False # Verbose, defined on top
    l.maximize = True # Maximize the fitness function
    if graph:
        l.storeAllPopulations = True # Keep history
    l.populationSize = evo.population_size # Population
    l.maxLearningSteps = evo.generations_max # Generations
    return l

# Initialize ID
def initialize(*args, **kwargs):
    print("----- Starting optimization -----")
    runtime_ID = aux.set_runtime_ID()
    print("Runtime ID:",runtime_ID)
    return runtime_ID

############### MAIN ###############
runtime_ID = initialize() # Start up code and give a random runtime ID

states, neighbors = make_states(task['m'],task['max_neighbors'])
desired_states_idx = find_desired_states_idx(states)
Q0 = init_policy(states,desired_states_idx)
GS,s = GS_active(Q0, states)



# from networkx.drawing.nx_agraph import to_agraph
# A = to_agraph(GS)
# A.layout('dot')
# A.draw()
# nx.draw_planar()
# plot.show()
# M = 8
# ## Learning parameters
# x0 = np.ones(M)/2 # Initialize to ones
# GA.xBound = list(zip(list(np.zeros(M)),list(np.ones(M)))) # Set limits
# GA.elitism = True # Use elite mem
# GA.mutationProb = evo['mutation_rate']
# GA.verbose = True
# GA.mutationStdDev = 0.2
# learner = GA(objF, x0) # Set up GA (alternative subclass)
# learner, GA = initialize_evolution_parameters(learner,evo)
#
#
