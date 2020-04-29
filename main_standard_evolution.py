import random
import numpy as np
from tools import fileHandler as fh
import evolution
from simulator import swarmulator
import pickle

sim = swarmulator.swarmulator(verbose=False)
sim.make(controller="controller_aggregation", agent="particle", clean=True, logger=False, verbose=False)
sim.runtime_setting("time_limit", str("50"))
sim.runtime_setting("simulation_realtimefactor", str("50"))
sim.runtime_setting("environment", "square")

def fitness(individual):
	policy_file = "../swarmulator/conf/state_action_matrices/aggregation_policy_evolved.txt"
	fh.save_to_txt(individual, policy_file)
	sim.runtime_setting("policy", policy_file) # Use random policy
	f = []
	for i in range(5):
		r = random.randrange(5,15)
		f = np.append(f,sim.run(r,run_id=1))
		print(f)
	return f.astype(float).mean(), # comma to cast to tuple!

e = evolution.evolution()
e.setup(fitness, GENOME_LENGTH=8, POPULATION_SIZE=100)
p = e.evolve(verbose=True, generations=100)
e.save("evo_run")