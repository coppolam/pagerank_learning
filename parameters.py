#!/usr/bin/env python3
'''
Loads the appropriate settings for each task
@author: Mario Coppola, 2020
'''

def get(c):
	'''
	Loads the desired parameters used throughout this paper for different tasks.

	The tasks are called according to their controller name in swarmulator.
	Study Case A = "aggregation"
	Study Case B = "forage"
	Study Case C1 = "pfsm_exploration"
	Study Case C2 = "pfsm_exploration_mod"

	@author: Mario Coppola, 2020
	'''
	
	# Study Case A
	if c == "aggregation":
		fitness = "aggregation_clusters"
		controller = "aggregation"
		agent = "particle"
		pr_states = 8
		pr_actions = 1

	# Study Case C
	elif c == "forage":
		fitness = "food"
		controller = "forage"
		agent = "particle_oriented"
		pr_states = 30
		pr_actions = 1

	# Study Case B1
	elif c == "pfsm_exploration":
		fitness = "aggregation_clusters"
		controller = "pfsm_exploration"
		agent = "particle_oriented"
		pr_states = 16
		pr_actions = 8

	# Study Case B2
	elif c == "pfsm_exploration_mod":
		fitness = "aggregation_clusters"
		controller = "pfsm_exploration_mod"
		agent = "particle_oriented"
		pr_states = 16
		pr_actions = 8

	# Study Case A -- online learning
	elif c == "onlinelearning_aggregation":
		fitness = "aggregation_clusters"
		controller = "onlinelearning_aggregation"
		agent = "particle"
		pr_states = 8
		pr_actions = 1

	# Study Case B1 -- online learning
	elif c == "onlinelearning_pfsm":
		fitness = "aggregation_clusters"
		controller = "onlinelearning_pfsm"
		agent = "particle_oriented"
		pr_states = 16
		pr_actions = 8

	else:
		ValueError("Unknown controller!")

	# Returna tuple with all relevant data
	return fitness, controller, agent, pr_states, pr_actions