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
import deap
# Own libraries
import graph as gt
import networkx as nx
import random

# Top level settings
np.set_printoptions(suppress=True) # Prevent numpy exponential notation on print, default False

# Parameters of the consensus task
task = {
    'max_neighbors': 8, # Maximum neighbors that the agent can see
    'm': 3 # Number of choices
}

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

# Initialize the policy (state action matrix)
def init_policy(states, desired_states_idx):
    # Generate a state-action matrix with all equal probabilities
    P0 = np.full((np.size(states, 0), np.size(states, 1) - 1), 1.0 / (np.size(states, 1) - 1))
    # Set action probabilities of desired states to 0
    P0[desired_states_idx] = 0
    return P0

# Make the active graph GSa. This function needs to be customized for a given task.
def GS_active(Q, states):
    n_states = np.size(states, 0) # number of states
    n_choices = np.size(Q, 1) # number of choices
    s = [] # Starting nodes
    t = [] # End nodes
    w = [] # Weights
    for i in range(0, n_states): # For each state, we will extract the possible edges and weights
        s_i = states[i, :] # Selected state
        indexes = [] # Indexes
        for j in range(0, n_choices):
            indexes.extend(np.where((states == np.append(j,s_i[range(1,n_choices+1)])).all(axis=1))[0])
        s.extend(int(i) * np.ones((n_choices,),dtype=int)) # Starting nodes
        t.extend(indexes) # End nodes
        w.extend(Q[i, :]) # Weight (equal to the action probability)

    Ga = gt.make_digraph(s, t, w) # Make the digraph
    return Ga

# Make the passive graph GSp. This function needs to be customized for a given task.
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
        s.extend(int(i) * np.ones((np.size(idx_final),),dtype=int)) # Starting nodes
        t.extend(idx_final) # Ending nodes

    G = gt.make_digraph(s, t) # Make the digraph. All weights the same in this case.
    return G

# Normalize the rows of an array
def normalize_rows(x):
    row_sums = x.sum(axis=1)
    new_matrix = x / row_sums[:, np.newaxis]
    return new_matrix

# Define the fitness function to optimize for
def fitness_function(pr):
    f = np.mean(pr[desired_states_idx])/np.mean(pr)
    return f

# Objective function
def objF(x):
    # Generate the policy as per the iteration x
    Q = Q0 # Copy Q from template Q0
    Q[active_rows,:] = np.reshape(x, (np.size(Q0, 0) - np.size(desired_states_idx), np.size(Q0, 1)))
    Q[active_rows,:] = normalize_rows(Q[active_rows,:])

    # Generate the active graph Gsa and its adjacency matrix H
    GSa = GS_active(Q, states)
    H = nx.adjacency_matrix(GSa,nodelist=range(np.size(neighbors)), weight='weight').toarray().astype(float)

    # Get Google Matrix using the equation: G = alpha_mat * (H + D) + (1- alpha_mat) * E
    S = np.add(H,D) # S = H + D
    S = normalize_rows(S) # Normalize S = H + D
    GoogleMatrix = np.add(np.matmul(alpha_mat,S),np.matmul(np.eye(np.size(alpha_mat,0)) - alpha_mat,E))

    # Evaluate fitness
    pr = pagerank(GoogleMatrix) # Evaluate PageRank vector
    fitness = fitness_function(pr) # Evaluate the fitness
    print(fitness)
    return 1/fitness # Using 1/f because we minimize instead of maximizing

# Initialize ID
def initialize(*args, **kwargs):
    print("----- Starting optimization -----")
    runtime_ID = random.randrange(1000)
    print("Runtime ID:",runtime_ID)
    return runtime_ID

############### MAIN ###############
runtime_ID = initialize() # Start up code and give a random runtime ID

# Initialize the state space, indicate the desired states, and the basic policy
states, neighbors = make_states(task['m'],task['max_neighbors'])
desired_states_idx = find_desired_states_idx(states)
Q0 = init_policy(states,desired_states_idx)
Q_idx = np.where(Q0.flatten()!=0)

# Generate the passive graph
GSa = GS_active(Q0, states)
GSp = GS_passive(states, neighbors)

# Generate alpha vector
alpha_vector = np.divide(np.sum(Q0, axis=1), neighbors + 1)
alpha_mat = np.diag(alpha_vector)

# E and D matrices (constant)
E = nx.adjacency_matrix(GSp, nodelist=range(np.size(neighbors)), weight='weight').toarray().astype(float)
D = np.zeros([np.size(E, 0), np.size(E, 1)])
D[desired_states_idx, :] = E[desired_states_idx, :]
E = normalize_rows(E)

active_rows = np.setdiff1d(range(0, np.size(Q0, 0)), desired_states_idx)

# Print the graphs (mainly for debugging purposes)
gt.print_graph(GSa,'GS_active.png')
gt.print_graph(GSp,'GS_passive.png')
x0 = np.ones(np.size(Q_idx))/task['m'] # Initialize to ones

# Learning parameters
bounds = list(zip(list(np.zeros(np.size(Q_idx))),list(np.ones(np.size(Q_idx))))) # Set limits
from scipy.optimize import differential_evolution
l = sp.optimize.minimize(objF, x0, args=(), bounds=bounds, tol=1e-3) # disp=True,popsize=10) # Set up GA (alternative subclass)

# Final policy
Q = Q0 # Copy Q from template Q0
Q[active_rows,:] = np.reshape(l.x, (np.size(Q0, 0) - np.size(desired_states_idx), np.size(Q0, 1)))

print(Q)
