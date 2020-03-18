import numpy as np
import matrixOperations as matop
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import itertools
import simulator_backend as sb

np.random.seed(2)
n_min, n_max = 10, 20
verbose = 1
estimator = estimator()
m = sb.m

def episode(perms, policy):
	### Initialize
	n = np.asscalar(np.random.randint(n_min, n_max, 1))
	pattern = sb.generate_random_connected_pattern(n)
	if verbose > 0:
		print("Number of robots: " + str(n))
	# plt.plot(pattern.T[0],pattern.T[1], 'ro'), plt.show()
	choices = np.random.randint(0, m, n)
	happy = False
	steps = 0

	### Run the simulation until agreement system
	while not happy:
		choices = sb.take_action(perms, pattern, choices, policy)
		if np.unique(choices.astype("int")).size is 1:
			happy = True
			if verbose > 0:
				print("Done! Result = ["+str(choices)+"]")
		if verbose > 1:
			print(choices)
		steps += 1

	return steps

def fitness(steps):
	fitness = steps
	return fitness

def local_state(a):
	# TODO: change this method of getting perms, it's pretty inefficient
	perms = np.array([])
	for perm in itertools.product(a, repeat=m):
		perms = np.append(perms, perm, axis = 0)
	perms = np.reshape(perms, (np.size(perms) // m, m))
	perms = perms[np.sum(perms, axis=1) <= 1]
	perms = np.around(perms, 1)
	return perms

def simulate(policy, perms):
	policy = np.reshape(policy, (np.size(policy) // m, m))
	policy = matop.normalize_rows(policy)
	steps = episode(perms, policy)
	f = fitness(steps)
	return f

def main():
	a = np.arange(0, 1.01, 0.2)
	perms = local_state(a)
	policy = np.ones([np.size(perms)//m,m])/m
	
	# bounds = list(zip(np.zeros(np.size(policy)),np.ones(np.size(policy)))) # Bing policy between 0 and 1
	# result = spopt.differential_evolution(objective_function, bounds, args=(perms,))
	f = simulate(policy, perms)
	
	if verbose > 0:
		print("Fitness = " + str(f))

if __name__ == '__main__':
	main()
