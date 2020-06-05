#!/usr/bin/env python3
"""
Backend engine for the consensus simulator
@author: Mario Coppola, 2020
"""
import sys, itertools
import numpy as np
from tools import matrixOperations as matop
from simulators import estimator

class consensus_simulator:
	def __init__(self, n=10, m=2, d=0.2):
		self.n = n # Robots
		self.m = m # Choices 
		self.r = 1.8 # Sensing distance 
		
		# Initialize local states
		self.discretization = d # Local state space discretization
		a = np.arange(0, 1.01, self.discretization)
		self.perms = self._local_state(a, m)
		
		# Initialize policy
		self.policy = np.ones(self.perms.shape)/m
		self.reset(n)

		# Estimator
		self.e = estimator.estimator(0.1)
		self.e.set_size(self.perms.shape[0])
		
	def generate_random_connected_pattern(self,n):
		i = 1
		pattern = np.array([[0, 0]]) # First robot
		while i < n:
			# Get the new position
			p = np.random.randint(0, i) # Attach to another neighbor
			new_position = pattern[p,:] + np.random.randint(-1,1+1,size=(1,2))[0]
			
			# Check if the position is occupied (you are the only allowd to occupy it!)
			if np.size(np.where((pattern == new_position).all(axis=1))) < 1:
				# It's free! Add it to the pattern.
				pattern = np.vstack((pattern, new_position))
				pattern = np.around(pattern, 0)
				i += 1

		return pattern

	def get_neighbors(self,selected_robot, pattern, r):
		p = pattern[selected_robot] - pattern
		d = np.sqrt(p[:,0]**2 + p[:,1]**2)
		# np.delete(d,selected_robot) # Remove yourself!
		return np.where(d < r)

	def reset(self, n):
		self.pattern = tools.generate_random_connected_pattern(n)
		self.choices = np.random.randint(0, self.m, n)

	def run(self, policy=None):
		# Run until agreement system
		happy = False
		steps = 0
		if policy is not None:
			self.policy = policy 
		while not happy:
			c = self._take_action()
			# print(steps, self.choices)
			if np.unique(c.astype("int")).size is 1: 
			# if steps > 10000:
				happy = True
			else:
				steps += 1
		return steps
	
	## Private methods
	def _local_state(self,a,m):
    	# TODO: change this method of getting perms, it's pretty inefficient
		perms = np.array([])
		for perm in itertools.product(a, repeat=m):
			perms = np.append(perms, perm, axis = 0)
		perms = np.reshape(perms, (perms.size // m, m))
		perms = perms[np.sum(perms, axis=1) <= 1]
		perms = np.around(perms, 1)
		return perms

	def _get_observation(self, selected_robot, choices):
		neighbors = tools.get_neighbors(selected_robot, self.pattern, self.r)
		return choices[neighbors], neighbors[0]

	def _get_observation_idx(self, sensor):
		observation = matop.normalize_rows(np.bincount(sensor), axis=0)
		observation = matop.round_to_multiple(observation, self.discretization)
		observation = np.pad(observation,(0, self.m - observation.size))
		observation = np.around(observation,1)
		observation_idx = np.where((self.perms == observation).all(axis=1))
		return observation_idx

	def _take_action(self):
		selected_robot = np.asscalar(np.random.randint(0, self.pattern.shape[0], 1))
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

		s = np.zeros([1,self.n])[0].astype("int")
		for robot in range(0,self.n):
			sensor, neighbors = self._get_observation(robot, self.choices)
			s[robot] = np.asscalar(self._get_observation_idx(sensor)[0])
		
		fitness = np.bincount(s).max() / self.n
		self.e.updateF(fitness, s)
		return self.choices
