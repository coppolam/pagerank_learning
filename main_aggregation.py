#!/usr/bin/env python3
"""
Simulate the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""

import argparse, sys
import aggregation as env

tmax = 100
r = [30]
file = "data/1_learning_data_t%i_r%i.npz"

sim = env.aggregation()
# sim.make(sys.argv[1], sys.argv[2], animation=True)

# if sys.argv[1]=="pfsm_exploration":
policy = "conf/state_action_matrices/exploration_policy_random.txt"
# else: 
# 	policy = ""

for i in r:
	sim.run(time_limit=tmax, robots=i, environment="square", policy=policy, run_id=1)
	filename_ext = ("_t%i_r%i" % (tmax, i))
	sim.save_learning_data(filename_ext=filename_ext)
