import random, sys, pickle, argparse, os
import numpy as np
from tools import fileHandler as fh
import evolution
from simulators import swarmulator

## Run as
# python3 main_standard_evolution.py CONTROLLER AGENT
# Example:
# python3 main_standard_evolution.py aggregation particle

#####################
#  Argument parser  #
#####################
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('controller', type=str, help="Controller to use")
parser.add_argument('agent', type=str, help="Agent to use")
parser.add_argument('-gen', type=int, help="Max generations", default=100)
parser.add_argument('-batchsize', type=int, help="Batch size", default=3)
parser.add_argument('-resume', type=bool, help="Resume after quitting", default=False)
parser.add_argument('-plot', type=str, help="", default=None)
parser.add_argument('-id', type=int, help="Evo ID", default=1)
args = parser.parse_args()

folder = "data/evolution_%s_%s/" % (args.controller,args.agent)
directory = os.path.dirname(folder)
if not os.path.exists(directory): os.makedirs(directory)

######################
#  Fitness function  #
######################
def fitness(individual):
	### Set the policy file that swarmulator reads
	policy_file = "../swarmulator/conf/state_action_matrices/policy_evolved_temp.txt"
	if args.controller=="pfsm_exploration":
		individual = np.reshape(individual,(16,8))
	fh.save_to_txt(individual, policy_file)
	sim.runtime_setting("policy", policy_file) # Use random policy

	### Run swarmulator in batches
	f = sim.batch_run((10,20),args.batchsize) # Run with 10-20 agents, 5 times
	return f.mean(), # Fitness = average (note trailing comma to cast to tuple!)

########################
#  Load evolution API  #
########################

e = evolution.evolution()
e.setup(fitness, GENOME_LENGTH=8, POPULATION_SIZE=100)

### Plot file from file args.plot
if args.plot is not None:
	e.load(args.plot)
	e.plot_evolution()
	exit()
	
#####################
#  Swarmulator API  #
#####################
sim = swarmulator.swarmulator(verbose=False)
sim.make(controller=args.controller, agent=args.agent, clean=True, logger=False, verbose=False)
sim.runtime_setting("time_limit", str("100"))
sim.runtime_setting("simulation_realtimefactor", str("300"))
sim.runtime_setting("environment", "square")
filename = folder + "evo_run_%s_%s_%i" % (args.controller, args.agent, args.id)

if args.controller == "aggregation":
	fitness = "aggregation_clusters"
elif args.controller == "pfsm_exploration":
	fitness = "aggregation_clusters"
elif args.controller == "forage":
	fitness = "food"
else:
	ValueError("Uknown inputs!")

sim.runtime_setting("fitness", fitness)

### Resume evolution from file args.resume
if args.resume is True:
	e.load(args.resume)
	p = e.evolve(verbose=True, generations=args.gen, checkpoint=filename, population=e.pop)

### Just run normally
else:
    p = e.evolve(verbose=True, generations=args.gen, checkpoint=filename)

### Save
e.save(filename)
