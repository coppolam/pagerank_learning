import numpy as np
import simulator_tools as st

def get_observation(selected_robot, pattern, choices):
	neighbors = st.get_neighbors(selected_robot, pattern)
	return choices[neighbors], neighbors

class estimator:
	def __init__(self,increment):
		self.H = None
		self.A = None
		self.E = None
		self.inc = increment

	def set_size(self, a):
		self.H = np.zeros([a,a])
		self.A = np.zeros([a,a])
		self.E = np.zeros([a,a])

	def update(self, k0, k1, action, pattern, choices):
		self.H[k0,k1] += self.inc
		self.A[k0,k1] += action
		for robot in expression_list:
			get_observation(robot, pattern, choices)

		self.E[k0,k1] = get_observation(robot, pattern, choices)