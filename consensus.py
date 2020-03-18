import numpy as np
import math
import matrixOperations as matop
import itertools
import time
import matplotlib.pyplot as plt

np.random.seed(5)

n_min = 10
n_max = 20
m = 2
r = 2

def get_neighbors(selected_robot, pattern):
    p = pattern[selected_robot] - pattern
    d = np.sqrt(p[:,0]**2 + p[:,1]**2)
    # np.delete(d,selected_robot)
    return np.where(d < r)

def get_observation(selected_robot, pattern, choices):
    neighbors = get_neighbors(selected_robot, pattern)
    return choices[neighbors], neighbors

def generate_random_pattern(n):
    pattern = np.random.randint(-2,2,size=(n,2)) # TODO: Fix
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
    else:
        print("happy")
    return choices

def episode(perms,policy):
    # Initialize
    n = np.asscalar(np.random.randint(n_min, n_max, 1))
    pattern = generate_random_pattern(n)
    plt.plot(pattern.T[0],pattern.T[1], 'ro')
    plt.show()
    choices = np.random.randint(0,m,n)
    happy = False
    steps = 0

    # Run
    while not happy:
        choices = take_action(perms,pattern, choices, policy)
        if np.unique(choices.astype("int")).size is 1:
            happy = True
        print("Done! Choices = ["+str(choices)+"]")     
        # time.sleep(0.1)       
        print(steps)
        steps += 1

    return steps

def fitness(steps):
    fitness = steps

def evaluate(perms,policy):
    steps = episode(perms,policy)
    return fitness(steps)

def main():
    a = np.arange(0, 1.01, 0.2)
    perms = np.array([])
    s = 0
    # TODO: change this method of getting perms, it's pretty inefficient
    for perm in itertools.product(a,repeat=m):
        perms = np.append(perms,perm,axis=0)
        s += 1
    perms = np.reshape(perms,(np.size(perms)//m,m))
    perms = perms[np.sum(perms,axis=1)<=1]
    perms = np.around(perms,1)
    policy = np.ones([np.size(perms),m])/m
    # states = 
    f = evaluate(perms,policy)
    print(f)

if __name__ == '__main__':
	main()
