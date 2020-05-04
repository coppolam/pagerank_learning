#!/usr/bin/env python3
"""
Simulate the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""
rerun = True

import argparse
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx 
from tools import fitness_functions as f

import aggregation as env
sim = env.aggregation()
tlim = 10000
r = 10
inc = 5
if rerun:
	sim.make(controller="pfsm_exploration", agent="particle_oriented", animation=True)
	for i in range(1,inc+1):
		sim.run(run_id=1, time_limit=tlim, robots=r*i, environment="square",
		policy="conf/state_action_matrices/exploration_policy_random.txt")
		filename_ext = ("_t%i_r%i" % (tlim, r*i))
		sim.save_learning_data(filename_ext=filename_ext)
else:
	sim.load(sys.argv)
	# sim.sim.plot_log()

sim.disp()
# sim.optimize()