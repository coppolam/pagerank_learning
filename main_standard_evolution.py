import evolution
from simulator import swarmulator

sim = swarmulator.swarmulator()
sim.make()
sim.runtime_setting("time_limit", str("100"))

def fitness(individual):
	f = sim.run(10,run_id=0)
	return sum(individual), # comma to cast to tuple!

def constraint(individual):
    if sum(individual[0:3]) > 1:
        return False # Failed
    return True # Pass 

e = evolution.evolution()
e.setup(fitness,GENOME_LENGTH=8)
p = e.evolve(verbose=True,generations=50)
p = e.evolve(verbose=True, population=p) # Continue

e.plot_evolution()