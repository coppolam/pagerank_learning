#!/usr/bin/env python3
"""
Simulate the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""

import aggregation, sys
import numpy as np
rerun = False

sim = aggregation.aggregation()

if rerun:
	sim.run(time_limit=100)
	sim.save()
else:
	sim.load(sys.argv)
	# sim.sim.plot_log()

# distance_to_others()

def evaluate_fitness(d):
    # for r in sim.robots
	p = d[:,(2,3)]
	for r in range(0,sim.robots)
		p[r,:]-p
	return fitness

time_column = 0
t = np.unique(sim.data[:,0])
fitness = np.zeros(t.shape)
for step in t:
	d = sim.data[np.where(sim.data[:,time_column] == step)]
	fitness = evaluate_fitness(d)

# sim.optimize()
# sim.evaluate(runs=1)
# sim.histplots()