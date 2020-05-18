#!/usr/bin/env python3
"""
Simulate the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""

import argparse, sys
import aggregation as env

# Input argument parser
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('controller', type=str, help="Controller to use")
parser.add_argument('agent', type=str, help="Agent to use")
parser.add_argument('-t', type=int, help="Max time of simulation. Default = 10000s", default=10000)
parser.add_argument('-n', type=int, help="Size of swarm. Default = 30", default=30)
parser.add_argument('-animate', type=bool, help="Turn on/off animation", default=False)
parser.add_argument('-id', type=bool, help="Sim ID", default=1)
args = parser.parse_args()

sim = env.aggregation() # High level simulator interface, connected to swarmulator
sim.make(args.controller, args.agent, animation=args.animate) # Build

# Ad-hoc arguments
if args.controller=="pfsm_exploration":
	policy = "conf/state_action_matrices/exploration_policy_random.txt"
else: 
	policy = ""

sim.run(time_limit=args.t, robots=args.n, environment="square", policy=policy, run_id=args.id)
filename_ext = ("%s_%s_t%i_r%i_id%i" % (args.controller, args.agent, args.t, args.n, args.id))
sim.save_learning_data(filename_ext=filename_ext)