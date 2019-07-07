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

# TODO: Make a params file for all the graph ect things
graph = True
M = 8
folder = '../pagerank_learning/';

# Test
E = np.matlib.eye(M)
aM = np.matlib.eye(M,M) * 0.5 # Alpha vector in matrix format
H = np.matlib.ones((M,M)) * 0.2

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
