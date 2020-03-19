## Libraries
import numpy as np
np.set_printoptions(suppress=True)  # Prevent numpy exponential notation on print, default False
import matplotlib.pyplot as plot
import random
import networkx as nx
import graph as gt
from pybrain.optimization import GA

# Subclass of GA with own definitions
class GA_alt(GA):
    def copy(self):         return GA_alt(self.x)

def set_runtime_ID():
    r = random.randrange(1000)
    return r

def objF(x):
    s = (0, 0, 0, 0, 0, 0, 0 ,1 ,1 ,1 ,1 ,1 ,1, 2 ,2 ,2 ,2 ,2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6)
    t = (1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 3, 4 ,5 ,6, 7, 4, 5, 6, 7, 5, 6, 7, 6, 7, 7)
    elist = list(zip(s, t, x))
    G = nx.DiGraph()
    G.add_weighted_edges_from(elist)
    pr = nx.pagerank(G, weight='weight', alpha=0.2)
    pr_values = list(pr.values())
    fitness = pr_values[1]/np.mean(pr_values)
    return fitness

def extract_history(l):
    # Extract fitness history
    fitness_history = []
    for x in range(0, l.numLearningSteps):
       fitness_history.append(max(l._allGenerations[x][1]))
    return fitness_history

def initialize_evolution_parameters(l):
    l.verbose = True # Verbose, defined on top
    l.minimize = False # Maximize the fitness function
    l.storeAllPopulations = True # Keep history
    l.populationSize = 20 # Population
    l.maxLearningSteps = 1000 # Generations
    return l

runtime_ID = set_runtime_ID()
print("Starting Optimization \n Runtime ID", runtime_ID)

## Learning parameters
G = nx.complete_graph(8)
n_edges = len(G.edges)
x0 = np.ones(n_edges) # Initialize to ones
lim = list(zip(list(np.zeros(n_edges)),list(np.ones(n_edges)))) # Bind values
GA.xBound = lim # Set limits, this is not from standard version of GA!!!
GA.elitism = True # Use elite mem
l = GA(objF, x0) # Set up GA (alternative subclass)
l = initialize_evolution_parameters(l)

## Learn
l.learn()

## Evaluate output
fitness_history = extract_history(l)
print(l.bestEvaluable)
plot.plot(fitness_history)
plot.show()