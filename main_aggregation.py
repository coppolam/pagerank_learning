#!/usr/bin/env python3
"""
Simulate the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""

import argparse, sys
import aggregation as env

tmax = 10000
r = [10,20,30]
file = "data/1_learning_data_t%i_r%i.npz"

sim = env.aggregation()

# sim.make(controller="controller_aggregation", agent="particle")
sim.make(controller="pfsm_exploration", agent="particle_oriented")

for i in r:
	sim.run(time_limit=tmax, robots=i, environment="square",
	policy="conf/state_action_matrices/exploration_policy_random.txt")
	filename_ext = ("_t%i_r%i" % (tmax, i))
	sim.save_learning_data(filename_ext=filename_ext)
