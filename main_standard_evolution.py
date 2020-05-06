import random, sys, pickle
import numpy as np
from tools import fileHandler as fh
import evolution
from simulator import swarmulator

c = sys.argv[1]
a = sys.argv[2]

sim = swarmulator.swarmulator(verbose=False)
sim.make(controller=c, agent=a, clean=True, logger=False, verbose=False)
sim.runtime_setting("time_limit", str("200"))
sim.runtime_setting("simulation_realtimefactor", str("50"))
sim.runtime_setting("environment", "square")
filename = "evo_run_%s_%s" % (c, a)

def fitness(individual):
	policy_file = "../swarmulator/conf/state_action_matrices/policy_evolved_temp.txt"
	individual = np.reshape(individual,(16,8))
	fh.save_to_txt(individual, policy_file)
	sim.runtime_setting("policy", policy_file) # Use random policy
	f = sim.batch_run((10,30),5)
	print(f)
	return f.mean(), # comma to cast to tuple!

e = evolution.evolution()
e.setup(fitness, GENOME_LENGTH=16*8, POPULATION_SIZE=100)
e.load(filename)
p = e.evolve(verbose=True, generations=100, checkpoint=filename, population=e.pop)
e.save(filename)
