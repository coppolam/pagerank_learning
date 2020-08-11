#!/usr/bin/env python3
"""
Loads the desired parameters used throughout this paper for different tasks.

The tasks are called according to their controller name in swarmulator.
Study Case A = "aggregation"
Study Case B = "forage"
Study Case C1 = "pfsm_exploration"
Study Case C2 = "pfsm_exploration_mod"

@author: Mario Coppola, 2020
"""

def get(c):
	if c == "aggregation": # Study Case A
		fitness = "aggregation_clusters"
		controller = "aggregation"
		agent = "particle"
		pr_states = 8
		pr_actions = 1
	elif c == "forage": # Study Case B
		fitness = "food"
		controller = "forage"
		agent = "particle_oriented"
		pr_states = 30
		pr_actions = 1
	elif c == "pfsm_exploration": # Study Case C1
		fitness = "aggregation_clusters"
		controller = "pfsm_exploration"
		agent = "particle_oriented"
		pr_states = 16
		pr_actions = 8
	elif c == "pfsm_exploration_mod": # Study Case C2
		fitness = "aggregation_clusters"
		controller = "pfsm_exploration_mod"
		agent = "particle_oriented"
		pr_states = 16
		pr_actions = 8
	else:
		ValueError("Unknown controller!")

	return fitness, controller, agent, pr_states, pr_actions