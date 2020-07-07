def get(c):
	if c == "aggregation":
		fitness = "aggregation_clusters"
		controller = "aggregation"
		agent = "particle"
		pr_states = 8
		pr_actions = 1
	elif c == "dispersion":
		fitness = "dispersion_clusters"
		controller = "aggregation"
		agent = "particle"
		pr_states = 8
		pr_actions = 1
	elif c == "pfsm_exploration":
		fitness = "aggregation_clusters"
		controller = "pfsm_exploration"
		agent = "particle_oriented"
		pr_states = 16
		pr_actions = 8
	elif c == "pfsm_dispersion":
		fitness = "dispersion_clusters"
		agent = "particle_oriented"
		controller = "pfsm_exploration"
		pr_states = 16
		pr_actions = 8
	elif c == "pfsm_exploration_mod":
		fitness = "aggregation_clusters"
		controller = "pfsm_exploration_mod"
		agent = "particle_oriented"
		pr_states = 16
		pr_actions = 8
	elif c == "forage":
		fitness = "food"
		controller = "forage"
		agent = "particle_oriented"
		pr_states = 30
		pr_actions = 1
	else:
		ValueError("Unknown controller!")

	return fitness, controller, agent, pr_states, pr_actions