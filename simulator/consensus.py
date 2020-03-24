#!/usr/bin/env python3
"""
Backend engine for the consensus simulator
@author: Mario Coppola, 2020
"""
import itertools
import numpy as np
from tools import matrixOperations as matop
from simulator import tools, estimator

class consensus_simulator:
	def __init__(self, n=10, m=2, d=0.2):
		self.n = n # Robots
		self.r = 1.8 # Sensing distance 
		self.m = m # Choices 
		self.discretization = d # Local state space discretization
		# Initialize local states
		a = np.arange(0, 1.01, self.discretization)
		self.perms = self._local_state(a, m)
		# Initialize policy
		policy = np.ones([np.size(self.perms)//m,m])/m
		policy = np.reshape(policy, (np.size(policy) // m, m))
		self.policy = matop.normalize_rows(policy)
		self.reset(n)

		# Estimator
		self.e = estimator.estimator(0.1)
		self.e.set_size(np.size(self.perms,0))
		
	def reset(self, n):
		self.pattern = tools.generate_random_connected_pattern(n)
		self.choices = np.random.randint(0, self.m, n)

	def run(self):
		# Run until agreement system
		happy = False
		steps = 0
		while not happy:
			c = self._take_action()
			# print(steps, self.choices)
			if np.unique(c.astype("int")).size is 1: 
			# if steps > 10000:
				happy = True
				print("Done! Steps = ["+str(steps)+"]")
			else:
				steps += 1
		return steps
	
	## Private methods
	def _local_state(self,a,m):
    	# TODO: change this method of getting perms, it's pretty inefficient
		perms = np.array([])
		for perm in itertools.product(a, repeat=m):
			perms = np.append(perms, perm, axis = 0)
		perms = np.reshape(perms, (np.size(perms) // m, m))
		perms = perms[np.sum(perms, axis=1) <= 1]
		perms = np.around(perms, 1)
		return perms

	def _get_observation(self, selected_robot, choices):
		neighbors = tools.get_neighbors(selected_robot, self.pattern, self.r)
		return choices[neighbors], neighbors[0]

	def _get_observation_idx(self, sensor):
		observation = matop.normalize_rows(np.bincount(sensor), axis=0)
		observation = matop.round_to_multiple(observation, self.discretization)
		observation = np.pad(observation,(0, self.m - np.size(observation)))
		observation = np.around(observation,1)
		observation_idx = np.where((self.perms == observation).all(axis=1))
		return observation_idx

	def _take_action(self):
		selected_robot = np.asscalar(np.random.randint(0, np.size(self.pattern,0), 1))
		sensor, neighbors = self._get_observation(selected_robot, self.choices)
		observation_idx = self._get_observation_idx(sensor)
		choices_old = np.copy(self.choices)

		# If in an active state, take an action
		if np.unique(sensor.astype("int")).size is not 1:
			action = np.random.choice(np.arange(0, self.m), 
						p=self.policy[np.asscalar(observation_idx[0])])
			self.choices[selected_robot] = action
			
		# Update H, A, E
		sensor, neighbors = self._get_observation(selected_robot, self.choices)
		observation_idx_new = self._get_observation_idx(sensor)
		self.e.updateH(observation_idx, observation_idx_new, self.choices[selected_robot]+1)
		
		for robot in neighbors:
			sensor, neighbors = self._get_observation(robot, choices_old)
			observation_idx = self._get_observation_idx(sensor)
			sensor_new, neighbors = self._get_observation(robot, self.choices)
			observation_idx_new = self._get_observation_idx(sensor_new)
			self.e.updateE(observation_idx, observation_idx_new)

		s = np.zeros([1,self.n])[0]
		for robot in range(0,self.n):
			sensor, neighbors = self._get_observation(robot, self.choices)
			s[robot] = np.asscalar(self._get_observation_idx(sensor)[0])
		
		s = s.astype("int")
		fitness = np.max(np.bincount(s))/self.n
		self.e.updateF(fitness, s)
		return self.choices