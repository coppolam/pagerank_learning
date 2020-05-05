#!/usr/bin/env python3
"""
Simulate the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""
rerun = False

import argparse, sys
import aggregation as env

tmax = 10000
r = 10
file = "data/1_learning_data_t%i_r%i.npz"
inc = 5

sim = env.aggregation()
if rerun:
	# sim.make(controller="controller_aggregation", agent="particle")
	sim.make(controller="pfsm_exploration", agent="particle_oriented")

	for i in range(1,inc+1):
		sim.run(time_limit=tmax, robots=r*i, environment="square",
		policy="conf/state_action_matrices/exploration_policy_random.txt")
		filename_ext = ("_t%i_r%i" % (tmax, r*i))
		sim.save_learning_data(filename_ext=filename_ext)
else:
	sim.load(file=(file %(tmax,r)))

# sim.disp()
# sim.optimize()

# sim.benchmark(time_limit=200,
# 	robots=10,
# 	runs=10,
# 	controller="pfsm_exploration", 
# 	agent="particle_oriented",
# 	policy="conf/state_action_matrices/exploration_policy_random.txt")

sim.histplots()