import random, sys, pickle
import numpy as np
from tools import fileHandler as fh
import evolution
from simulator import swarmulator
import argparse

parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('controller', type=str, help="Controller to use")
parser.add_argument('agent', type=str, help="Agent to use")
parser.add_argument('-gen', type=int, help="Max generations", default=100)
parser.add_argument('-batchsize', type=int, help="Batch size", default=10)
parser.add_argument('-resume', type=bool, help="Resume after quitting", default=False)
args = parser.parse_args()

sim = swarmulator.swarmulator(verbose=False)
sim.make(controller=args.controller, agent=args.agent, clean=True, logger=False, verbose=False)
sim.runtime_setting("time_limit", str("100"))
sim.runtime_setting("simulation_realtimefactor", str("0"))
sim.runtime_setting("environment", "square")
filename = "evo_run_%s_%s" % (args.controller, args.agent)

def fitness(individual):
	policy_file = "../swarmulator/conf/state_action_matrices/policy_evolved_temp.txt"
	if args.controller=="pfsm_exploration":
		individual = np.reshape(individual,(16,8))
	fh.save_to_txt(individual, policy_file)
	sim.runtime_setting("policy", policy_file) # Use random policy
	f = sim.batch_run((10,20),5)
	return f.mean(), # comma to cast to tuple!

e = evolution.evolution()
e.setup(fitness, GENOME_LENGTH=8, POPULATION_SIZE=100)
#e.plot_evolution()

if args.resume:
	e.load(filename)
	p = e.evolve(verbose=True, generations=args.gen, checkpoint=filename, population=e.pop)
else:
	p = e.evolve(verbose=True, generations=args.gen, checkpoint=filename)

e.save(filename)
