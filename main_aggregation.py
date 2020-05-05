#!/usr/bin/env python3
"""
Simulate the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""
rerun = True

import argparse, sys
import aggregation as env

tlim = 10000
r = 10
inc = 5

sim = env.aggregation()
if rerun:
	# sim.make(controller="controller_aggregation", agent="particle")
	sim.make(controller="pfsm_exploration", agent="particle_oriented")

	for i in range(1,inc+1):
		sim.run(run_id=1, time_limit=tlim, robots=r*i, environment="square",
		policy="conf/state_action_matrices/exploration_policy_random.txt")
		filename_ext = ("_t%i_r%i" % (tlim, r*i))
		sim.save_learning_data(filename_ext=filename_ext)
else:
	sim.load(sys.argv)

sim.disp()
# sim.optimize()
