import estimator as est
import matrixOperations as matop
import numpy as np

m = 2
r = 1.8

def get_neighbors(selected_robot, pattern):
	p = pattern[selected_robot] - pattern
	d = np.sqrt(p[:,0]**2 + p[:,1]**2)
	np.delete(d,selected_robot) # Remove yourself!
	return np.where(d < r)

def generate_random_connected_pattern(n):
	i = 1
	pattern = np.array([[0,0]]) # First robot
	while i < n:
		# Get the new position
		p = np.random.randint(0,i) # Attach to another neighbor
		new_position = pattern[p,:] + np.random.randint(-1,1+1,size=(1,2))[0]
		
		# Check if the position is occupied (you are the only allowd to occupy it!)
		if np.size(np.where((pattern == new_position).all(axis=1))) < 1:
			# It's free! Add it to the pattern.
			pattern = np.vstack((pattern, new_position))
			pattern = np.around(pattern, 0)
			i += 1
	return pattern

def take_action(perms, pattern, choices, policy):
	selected_robot = np.asscalar(np.random.randint(0,np.size(pattern,0),1))
	sensor, neighbors = est.get_observation(selected_robot, pattern, choices)
	sensor_prev = sensor
	if np.unique(sensor.astype("int")).size is not 1:
		observation = matop.normalize_rows(np.bincount(sensor),axis=0)
		observation = matop.round_to_multiple(observation, 0.2)
		observation = np.pad(observation,(0,m-np.size(observation)))
		observation = np.around(observation,1)
		observation_idx = np.where((perms == observation).all(axis=1))	
		action = np.random.choice(np.arange(0,m), p=policy[np.asscalar(observation_idx[0])])
		choices[selected_robot] = action

	# Update H, A, E
	e = est.estimator(0.1)
	e.update()
	
	return choices