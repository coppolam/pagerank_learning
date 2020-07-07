#!/usr/bin/env python3
"""
Optimize a behavior based on the PageRank function
@author: Mario Coppola, 2020
"""
import sys
import scipy.optimize
import numpy as np
from tools import matrixOperations as matop
np.set_printoptions(suppress=True) # Avoid scientific notation
from scipy.special import softmax
from classes import evolution

class pagerank_evolve:
	def __init__(self,des,A,E):
		# Calculate estimated alpha using ratio of H to E for each row
		## Solve for alpha as follows
		## Use: r = H/E = alpha/(1-alpha) (based on alpha = probability of H, (1-alpha) = probability of E)
		## Then: solve for alpha --> alpha = H/E / (1+H/E)
		H = np.sum(A,axis=0)
		with np.errstate(divide='ignore',invalid='ignore'):
			r = H.sum(axis=1) / E.sum(axis=1)
		r = np.nan_to_num(r) # Remove NaN Just in case
		self.alpha = r / (1 + r)
		# Normalize
		self.A = np.copy(A) # Doing this to avoid rewriting A
		for i in range(0,A.shape[0]): self.A[i] = matop.normalize_rows(self.A[i])
		self.E = matop.normalize_rows(E)
		self.des = des

	def _update_b(self,pol):
		b = np.zeros(self.A[0].shape)
		for i, p in enumerate(pol.T): # Iterate over each action (columns of pol)
			b += self.A[i] * p[:,np.newaxis] # [:,np.newaxis] makes p vertical
		# Matrix b holds the cumulative probability of the transition happening for a current policy
		# This is the probability of the action being taken times the 
		# probability of the state transition caused by the action, and then the sum of that
		# For example:
		# b[0,0] = P((e00 and a00) or (e00 and a1) or ... or (e00 and aN))
		#        = P(e00|a0)*P(a0) +  P(e00|a1)*P(a1) + ... + P(e00|aN)*P(aN))
		# where e00 is a state transition from state 0 to state 0
		# and a0... aN are the actions 0 to N
		# In essence b[0,0] = P(e00), given that the actions are independent at the local level
		return b

	def _update_H(self, pol):	
		# Reshape policy vector to the right matrix dimensions
		cols = self.A.shape[0]
		pol = np.reshape(pol,(len(pol)//cols,cols)) # Resize policy
		if cols > 1: pol = matop.normalize_rows(pol+0.001)
		
		# Update model based on new policy
		return self._update_b(pol)

	def _pagerank_fitness(self,pr):
		return np.average(pr,axis=1,weights=self.des)/pr.mean()

	def _fitness(self, policy):
		H1 = self._update_H(policy) # Update H with new policy
		G = np.diag(self.alpha).dot(H1) + np.diag(1-self.alpha).dot(self.E) # Google matrix
		pr = matop.pagerank(G) # Evaluate pagerank vector 
		f = self._pagerank_fitness(pr) # Get fitness
		return -f # Trick scipy into maximizing

	def _optimize(self, policy):
		e = evolution.evolution()
		e.setup(self._fitness, GENOME_LENGTH=policy.size, POPULATION_SIZE=20)
		e.evolve(verbose=False, generations=1000, population=None)
		e.plot_evolution()
		return e.get_best()

	def main(self, policy0):
		# Optimize using pagerank fitness
		policy = self._optimize(policy0)

		# Reformat output
		cols = policy0.shape[1] # Number of columns (as per input policy)
		policy = np.reshape(policy,(len(policy)//cols,cols)) # Resize pol
		if cols > 1: policy = matop.normalize_rows(policy) # Normalize rows
		
		return policy