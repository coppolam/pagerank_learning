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

# Evaluate pagerank vector
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
    return Ga

def GS_passive(states, neighbors):
    n_states = np.size(states, 0)
    s = [] # Starting nodes
    t = [] # End nodes
    for i in range(0, n_states): # For each state, we will extract the possible edges and weights
        s_i = states[i, :] # Selected state
        n_eq_idx = np.where(neighbors == neighbors[i])[0] # Indexes of states with the same number of neighbors

        # Extract indexes where there is only at most one neighbour changing states at the same time
        idx = np.where( np.amax(np.abs(np.subtract(states[n_eq_idx, 1:np.size(states, 1)], s_i[1:np.size(states, 1)])),axis=1) == True)
        n_eq_idx = n_eq_idx[idx]

        # Exclude your own state from there
        current_opinion_idx = np.where(states[:, 0] == s_i[0])
        idx_final = np.intersect1d(n_eq_idx,current_opinion_idx)

        # Create edges between starting nodes and end nodes
        s.extend(int(i) * np.ones((np.size(idx_final),),dtype=int))
        t.extend(idx_final)

    G = gt.make_digraph(s, t) # Make the digraph. All weights the same in this case.
    return G

def normalize_rows(x: np.ndarray,rows_of_interest=None):
    if rows_of_interest is None:
        rows_of_interest = range(np.size(x,0))

    row_sums = x.sum(axis=1)
    new_matrix = x
    new_matrix[rows_of_interest][:] = x[rows_of_interest][:] / row_sums[rows_of_interest, np.newaxis]
    return new_matrix

# Define the fitness function to optimize for
def fitness_function(pr):
    f = np.mean(pr[desired_states_idx])/np.mean(pr)
    return f

def objF(x):
    # Generate policy
    Q = Q0
    active_rows = np.setdiff1d(range(0, np.size(Q0,0)), desired_states_idx)
    Q[active_rows][:] = np.reshape(x, (np.size(Q0, 0) - np.size(desired_states_idx), np.size(Q0, 1)))

    # Generate alpha vector
    alpha_vector = np.divide(np.sum(Q,axis = 1),neighbors+1)
    alpha_mat = np.diag(alpha_vector)

    # Generate the graphs
    GSA = GS_active(Q, states)
    GSP = GS_passive(states, neighbors)

    # Print the graphs (mainly for debugging purposes)
    gt.print_graph(GSA, 'GS_active_debug.png')
    gt.print_graph(GSP, 'GS_passive_debug.png')

    # Adjacency Matrices, ensuring that the order is correct!
    H = nx.adjacency_matrix(GSA,nodelist=range(np.size(neighbors)), weight='weight').toarray()
    E = nx.adjacency_matrix(GSP,nodelist=range(np.size(neighbors)), weight='weight').toarray()

    H = normalize_rows(H,active_rows)
    E = normalize_rows(E)
    D = np.zeros([np.size(E,0),np.size(E,1)])
    D[desired_states_idx,:] = E[desired_states_idx,:]

    # Get Google Matrix  --->  G = alpha_mat * (H + D) + (1- alpha_mat) * E
    S = np.add(H,D)
    GoogleMatrix = np.add(np.matmul(alpha_mat,S),np.matmul(np.eye(np.size(alpha_mat,0)) - alpha_mat,E))

    pr = pagerank(GoogleMatrix) # Evaluate PageRank vector
    fitness = fitness_function(pr) # Evaluate the fitness
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
    l.storeAllPopulations = True # Keep history
    l.populationSize = evo['population_size'] # Population
    l.maxLearningSteps = evo['generations_max'] # Generations
    return l

# Initialize ID
def initialize(*args, **kwargs):
    print("----- Starting optimization -----")
    runtime_ID = aux.set_runtime_ID()
    print("Runtime ID:",runtime_ID)
    return runtime_ID

############### MAIN ###############
runtime_ID = initialize() # Start up code and give a random runtime ID

# Initialize the state space, indicate the desired states, and the basic policy
states, neighbors = make_states(task['m'],task['max_neighbors'])
desired_states_idx = find_desired_states_idx(states)
Q0 = init_policy(states,desired_states_idx)
Q_idx = np.where(Q0.flatten()!=0)

# Generate the graphs
GSA = GS_active(Q0, states)
GSP = GS_passive(states, neighbors)

# Print the graphs (mainly for debugging purposes)
gt.print_graph(GSA,'GS_active.png')
gt.print_graph(GSP,'GS_passive.png')

# Learning parameters
GA.xBound = list(zip(list(np.zeros(np.size(Q_idx))),list(np.ones(np.size(Q_idx))))) # Set limits
GA.elitism = True # Use elite mem
GA.mutationProb = evo['mutation_rate']
GA.verbose = True
GA.mutationStdDev = 0.2

x_init = np.ones(np.size(Q_idx))/task['m'] # Initialize to ones
l = GA(objF, x_init) # Set up GA (alternative subclass)
l = initialize_evolution_parameters(l,evo)

# Learn
l.learn()
