#!/usr/bin/env python3
"""
Backend engine for simulators
@author: Mario Coppola, 2020
"""
import numpy as np

def generate_random_connected_pattern(n):
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

def get_neighbors(selected_robot, pattern, r):
	p = pattern[selected_robot] - pattern
	d = np.sqrt(p[:,0]**2 + p[:,1]**2)
	# np.delete(d,selected_robot) # Remove yourself!
	return np.where(d < r)