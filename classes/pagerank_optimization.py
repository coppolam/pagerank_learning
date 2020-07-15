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

def reshape_policy(A, pol):
	'''Reshape the stochastic policy to the correct dimensions'''
	cols = A.shape[0]
	pol = np.reshape(pol,(pol.size//cols,cols)) # Resize policy
	if cols > 1: pol = matop.normalize_rows(pol)
	return pol
	
def update_H(A, policy):
	'''Update the H matrix for the chosen policy'''
	policy = reshape_policy(A, policy) # Ensure policy has the correct dimensions

	# Update H matrix
	## Matrix H1 holds the cumulative probability of the transition happening for a current policy
	## This is the probability of the action being taken times the 
	## probability of the state transition caused by the action, and then the sum of that
	## For example:
	## H[0,0] = P((e00 and a00) or (e00 and a1) or ... or (e00 and aN))
	##         = P(e00|a0)*P(a0) +  P(e00|a1)*P(a1) + ... + P(e00|aN)*P(aN))
	## where e00 is a state transition from state 0 to state 0
	## and a0... aN are the actions 0 to N
	## In essence H[0,0] = P(e00), given that the actions are independent at the local level
	H = np.zeros(A[0].shape)
	for i, p in enumerate(policy.T): # Iterate over each action (columns of pol)
		H += A[i] * p[:,np.newaxis] # [:,np.newaxis] makes p vertical
	if A.shape[0] > 1: H = matop.normalize_rows(H) # Normalize for multiple actions
	return H

def pagerankfitness(pr,des):
	return np.average(pr,axis=1,weights=des)/pr.mean() # Get pagerank fitness

def objective_function(pol, des, alpha, A, E):
	'''Objective function'''
	H1 = update_H(A, pol) # Update H with new policy
	G = np.diag(alpha).dot(H1) + np.diag(1-alpha).dot(E) # Google matrix
	pr = matop.pagerank(G) # Evaluate pagerank vector 
	f = pagerankfitness(pr,des)
	
	# Display progress to terminal
	p = "\r Fitness \t max:f=%.10f" % -f
	sys.stdout.write(p)
	sys.stdout.flush()

	return -f # Negative to trick scipy into maximizing

def optimize(pol0, des, alpha, A, E):
	'''Optimization function'''
	ll = 0.0 # Lower limit
	ul = 1.0 # Upper limit
	bounds = list(zip(ll*np.ones(pol0.size),ul*np.ones(pol0.size))) # Bind values
	return scipy.optimize.minimize(objective_function, pol0, bounds=bounds, args=(des, alpha, A, E))

def main(pol0, des, A, E):
	'''Main function to optimize the policy'''
	# Calculate estimated alpha using ratio of H to E for each row
	## Solve for alpha as follows
	## Use: r = H/E = alpha/(1-alpha) 
	## (based on alpha = probability of H, (1-alpha) = probability of E)
	## Then: solve for alpha --> alpha = H/E / (1+H/E)
	H = np.sum(A,axis=0)
	with np.errstate(divide='ignore',invalid='ignore'):
		r = H.sum(axis=1) / E.sum(axis=1)
	r = np.nan_to_num(r) # Remove NaN Just in case
	alpha = r / (1 + r)

	# Normalize
	An = np.copy(A) # Doing this to avoid rewriting A
	for i in range(0,A.shape[0]): An[i] = matop.normalize_rows(An[i])
	E = matop.normalize_rows(E)

	# Optimize using pagerank fitness
	result = optimize(pol0, des, alpha, An, E)
	return reshape_policy(An, result.x)