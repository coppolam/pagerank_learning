#!/usr/bin/env python3
"""
Simulate the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""

import argparse, sys
import aggregation as env

###########################
#  Input argument parser  #
###########################
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('controller', type=str, help="Controller to use")
parser.add_argument('agent', type=str, help="Agent to use")
parser.add_argument('-t', type=int, help="Max time of simulation. Default = 10000s", default=10000)
parser.add_argument('-n', type=int, help="Size of swarm. Default = 30", default=30)
parser.add_argument('-animate', type=bool, help="Turn on/off animation", default=False)
parser.add_argument('-id', type=int, help="Sim ID", default=None)
args = parser.parse_args()

#########################################
#  Ad-hoc settings for each controller  #
#########################################
if args.controller == "controller_aggregation":
	policy = ""
	pr_states = 8
	pr_actions = 1
elif args.controller == "pfsm_exploration":
	policy = "conf/state_action_matrices/exploration_policy_random.txt"
	fitness = "aggregation_clusters"
	pr_states = 16
	pr_actions = 8
elif args.controller == "forage":
	policy = ""
	pr_states = 16
	pr_actions = 1
else:
    ValueError("Unknown inputs!")
	
############
#   RUN    #
############
# Load and build
sim = env.aggregation(savefolder="data/learning_data_%s_%s/"%(args.controller,args.agent)) # Load high level simulator interface, connected to swarmulator
sim.make(args.controller, args.agent, animation=args.animate) # Build

# Run
sim.run(time_limit=args.t, robots=args.n, environment="square", policy=policy, 
	fitness=fitness, pr_states=pr_states, pr_actions=pr_actions, run_id=args.id)

# Save data
filename_ext = ("%s_%s_t%i_r%i_id%s" % (args.controller, args.agent, args.t, args.n, sim.run_id))
sim.save_learning_data(filename_ext=filename_ext)
