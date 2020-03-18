import numpy as np

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

	def updateH(self, k0, k1, action):
		self.H[k0,k1] += self.inc
		self.A[k0,k1] = action

	def updateE(self,k0,k1):
		self.E[k0,k1] += self.inc 