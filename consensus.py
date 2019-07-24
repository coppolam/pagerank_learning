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
import fitness_function as opt
import graph as graphtools
import scipy as sp
import itertools as tools
list(tools.permutations([1, 2, 3]))

graph = True

# import consensus_task as task
# TODO: Put all function in a "consensus class" such that all you need to do is import a different thing and everything changes
# This would be nicer because it would allow to keep one general main file where you just need to switch out the import
# This would also include the parameters that are needed
#parameters
max_neighbors = 8
n_choices = 2
folder = '../pagerank_learning/'

def make_states(n_choices,max_neighbors):
    max_neighbors = 8
    n_choices = 2
    ext_states = np.array(list(tools.product(range(0,max_neighbors+1),repeat=n_choices)))
    ext_states = ext_states[np.where(sp.sum(ext_states,1)<=max_neighbors)[0]]
    int_states = np.array(sp.repeat([1,2],np.size(ext_states,0)))
    edge_list = np.array(list(zip(np.array(int_states),ext_states)))
    return states

def init_Q(states):
    n_states = np.size(states,0)
    n_choices = np.size(states,1) - 1
    Q = np.ones(n_states,n_choices)/n_choices;
    return Qa.

def get_Q_idx(Q,states):
    # TODO: Make
    return Qidx



# Global variables. Is it possible to move these to a dedicated subfile?
states = make_states(n_choices, max_neighbors)
Q = init_Q(states)
Qidx = get_Q_idx(Q, states)

## Learning parameters
x0 = np.ones(M)/2 # Initialize to ones
GA.xBound = list(zip(list(np.zeros(M)),list(np.ones(M)))) # Set limits
GA.elitism = True # Use elite mem
#GA.eliteProportion = 0.1
GA.mutationProb = 0.5
GA.verbose = True
GA.mutationStdDev = 0.2
l = GA(opt.objF, x0) # Set up GA (alternative subclass)
l = opt.initialize_evolution_parameters(l)

## Learn
l.learn()

## Evaluate output
if graph:
    print(l.bestEvaluable)
    fitness_history = opt.extract_history(l)
    plot.pyplot.plot(fitness_history)
