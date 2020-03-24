#!/usr/bin/env python3
"""
General method to estimate the H, A, and E matrices throughout a simulation
@author: Mario Coppola, 2020
"""
import numpy as np

class estimator:
	def __init__(self, increment):
		self.H = None
		self.A = None
		self.E = None
		self.F = 0
		self.des = None
		self.inc = increment

	def set_size(self, a):
		self.H = np.zeros([a,a])
		self.A = np.zeros([a,a])
		self.E = np.zeros([a,a])

	def updateH(self, s0, s1, action):
		self.H[s0,s1] += self.inc
		self.A[s0,s1] = action

	def updateE(self, s0, s1):
		self.E[s0,s1] += self.inc

	def updateF(self, fitness, s):
		if fitness > self.F:
			self.F = fitness
			self.des = s
