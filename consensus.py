import numpy as np
import math
import matrixOperations as matop
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import itertools


np.random.seed(2)
n_min, n_max = 10, 20
m = 2
r = 1.8

def get_neighbors(selected_robot, pattern):
	p = pattern[selected_robot] - pattern
	d = np.sqrt(p[:,0]**2 + p[:,1]**2)
	np.delete(d,selected_robot) # Remove yourself!
	return np.where(d < r)

def get_observation(selected_robot, pattern, choices):
	neighbors = get_neighbors(selected_robot, pattern)
	return choices[neighbors], neighbors

def generate_random_connected_pattern(n):
	i = 1
	pattern = np.array([[0,0]]) # First robot
	while i < n:
		# Get the new position
		p = np.random.randint(0,i) # Attach to another neighbor
		new_position = pattern[p,:] + np.random.randint(-1,1+1,size=(1,2))[0]
		
		# Check if the position is occupied (you are the only allowd to occupy it!)
		if np.size(np.where((pattern == new_position).all(axis=1))) < 1:
			# Add it to the pattern
			pattern = np.vstack((pattern, new_position))
			pattern = np.around(pattern, 0)
			i += 1	

	return pattern

def take_action(perms, pattern, choices, policy):
	selected_robot = np.asscalar(np.random.randint(0,np.size(pattern,0),1))
	sensor, neighbors = get_observation(selected_robot, pattern, choices)
	if np.unique(sensor.astype("int")).size is not 1:
		observation = matop.normalize_rows(np.bincount(sensor),axis=0)
		observation = matop.round_to_multiple(observation, 0.2)
		observation = np.pad(observation,(0,m-np.size(observation)))
		observation = np.around(observation,1)
		observation_idx = np.where((perms == observation).all(axis=1))
		action = np.random.choice(np.arange(0,m), p=policy[np.asscalar(observation_idx[0])])
		choices[selected_robot] = action
	return choices

def episode(perms,policy):
	# Initialize
	n = np.asscalar(np.random.randint(n_min, n_max, 1))
	pattern = generate_random_connected_pattern(n)
	print(n)
	# plt.plot(pattern.T[0],pattern.T[1], 'ro'), plt.show()
	choices = np.random.randint(0, m, n)
	happy = False
	steps = 0

	# Run
	while not happy:
		choices = take_action(perms, pattern, choices, policy)
		if np.unique(choices.astype("int")).size is 1:
			happy = True
			print("Done! Result = ["+str(choices)+"]")
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

def objective_function(policy, perms):
	policy = np.reshape(policy, (np.size(policy) // m, m))
	policy = matop.normalize_rows(policy)
	steps = episode(perms, policy)
	f = fitness(steps)
	return f

def main():
	a = np.arange(0, 1.01, 0.2)
	perms = local_state(a)
	policy = np.ones([np.size(perms)//m,m])/m
	
	# Bound probabilistic policy to being between 0 and 1
	bounds = list(zip(np.zeros(np.size(policy)),np.ones(np.size(policy))))
	# Optimize
	result = spopt.differential_evolution(objective_function, bounds, args=(perms,))
	

if __name__ == '__main__':
	main()
