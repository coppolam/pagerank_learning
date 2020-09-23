#!/usr/bin/env python3
'''
Optimize a behavior based on the PageRank function

@author: Mario Coppola, 2020
'''

import numpy as np
from tools import matrixOperations as matop
from classes import evolution

class pagerank_evolve:
	'''Evolves a new policy based on the PageRank fitness'''

	def __init__(self,des,A,E):
		'''Initializer'''
		
    	# Initialize the evolution API
		self.e = evolution.evolution()

		# Calculate estimated alpha using ratio of H to E for each row
		## Solve for alpha as follows
		## Use: r = H/E = alpha/(1-alpha)
		## (based on alpha = probability of H, (1-alpha) = probability of E)
		## Then: solve for alpha --> alpha = H/E / (1+H/E)
		H = np.sum(A,axis=0)
		with np.errstate(divide='ignore',invalid='ignore'):
			r = H.sum(axis=1) / E.sum(axis=1)
		r = np.nan_to_num(r) # Remove NaN Just in case
		self.alpha = r / (1 + r)

		# Set the row normalized set of A matrices for each action
		self.A = np.copy(A) # copy to avoid rewriting A due to shallow copy
		for i in range(0,A.shape[0]):
			self.A[i] = matop.normalize_rows(self.A[i])

		# Set row normalized E matrix
		self.E = matop.normalize_rows(E)
		
		# Set desired states
		self.des = des

	def reshape_policy(self, A, policy):
		'''Reshape the stochastic policy to the correct dimensions'''

		# Get the number of columns and the policy as a numpy array
		cols = A.shape[0]
		policy = np.array(policy)

		# Resize policy
		policy = np.reshape(policy,(policy.size//cols,cols))
		
		# If more than 1 column, normalize rows
		if cols > 1:
			policy = matop.normalize_rows(policy)

		return policy
		
	def update_H(self, A, policy):
		'''
		Update the H matrix for the chosen policy
		
		Update H matrix
		Matrix H1 holds the cumulative probability of the transition 
		happening for a current policy.
		This is the probability of the action being taken times the 
		probability of the state transition caused by the action, 
		and then the sum of that.
		For example:
		H[0,0] = P((e00 and a00) or (e00 and a1) or ... or (e00 and aN))
				= P(e00|a0)*P(a0) +  P(e00|a1)*P(a1) + ... + P(e00|aN)*P(aN))
		where e00 is a state transition from state 0 to state 0
		and a0... aN are the actions 0 to N
		In essence H[0,0] = P(e00), given that the actions 
		are independent at the local level.
		'''
		# Ensure policy has the correct dimensions
		policy = self.reshape_policy(A, policy)

		## In this routine, we will iterate over each action (columns of policy)
		H = np.zeros(A[0].shape)
		for i, p in enumerate(policy.T):
			H += A[i] * p[:,np.newaxis] # [:,np.newaxis] makes p vertical

		# Normalize for multiple actions
		if A.shape[0] > 1:
			H = matop.normalize_rows(H)

		# Return updated H
		return H

	def pagerank_fitness(self,pr,des):
		'''
		Get the PageRank fitness.
		The fitness is the weighted average over the desired states
		'''
		return np.average(pr,axis=1,weights=des)/pr.mean()

	def _fitness(self, policy):
		'''Evaluate the fitness'''

		# Update H with new policy
		H1 = self.update_H(self.A,policy)

		# Google matrix
		G = np.diag(self.alpha).dot(H1) + np.diag(1-self.alpha).dot(self.E)
		
		# Evaluate pagerank vector
		pr = matop.pagerank(G)
		
		# Get fitness
		f = self.pagerank_fitness(pr,self.des)
		
		return f

	def _optimize(self, policy, generations=500, plot=False):
		'''Run the optimization'''

		# Set up the parameters
		self.e.setup(self._fitness, 
			GENOME_LENGTH=policy.size, POPULATION_SIZE=20)

		# Run the evolution
		self.e.evolve(verbose=True, generations=generations, population=None)
		
		# Plot
		# if plot:
		# 	self.e.plot_evolution()

		return self.e.get_best()

	def run(self, policy0, generations=500, plot=False):
		'''Get a policy and optimize it according to the PageRank scheme'''

		# Optimize using pagerank fitness
		policy = self._optimize(policy0,generations=generations, plot=plot)

		# Format and return the optimized policy
		return np.array(self.reshape_policy(self.A, policy))