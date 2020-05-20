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

def update_b(A,pol):
	b = np.zeros(A[0].shape)
	i = 0 # Iterator
	for p in pol.T: # Iterate over each action (columns of pol)
		b += A[i] * p[:,np.newaxis] # [:,np.newaxis] makes p vertical
		i += 1
	# Matrix b holds a measure of the probability of the transition happening 
	# given a policy, which is the probability of the action times the 
	# relative probability of the action leading to a particular state change.
	return b

def update_H(H0, b0, A ,E , pol0, pol):	
	# Reshape policy vector to the right matrix dimensions and normalize
	cols = pol0.shape[1]
	pol = np.reshape(pol,(pol.size//cols,cols)) # Resize policy
	if cols > 1: pol = matop.normalize_rows(pol+0.001) # Normalize +0.001 to keep connected

    ###########################################
	# Routine to update H based on new policy #
	###########################################
	# H1 = b1 / b0 * H0
	b1 = update_b(A, pol) # Use H0 since that holds the total # of transitions, we only change the weighing
	return np.divide(b1, b0, where=b0!=0) * H0;

def fitness(pr,des):
    return np.average(pr,axis=1,weights=des)/pr.mean()

def objective_function(pol, pol0, des, alpha, H, b0, A, E):
	H1 = update_H(H, b0, A, E, pol0, pol) # Update H with new policy
	G = np.diag(alpha).dot(H1) + np.diag(1-alpha).dot(E) # Google matrix
	pr = matop.pagerank(G) # Evaluate pagerank vector 
	f = fitness(pr, des) # Get fitness
	
	# Display to terminal
	p = "\r Fitness \t max:f=%.10f \t min:1/(1+f)=%.10f" % (f, 1/(f+1))
	sys.stdout.write(p)
	sys.stdout.flush()

	return 1 / (f + 1) # Trick scipy into maximizing

def optimize(pol0, des, alpha, H, b0, A, E):
	# Bind probabilistic policy
	ll = 0.0 # Lower limit
	up = 1.0 # Upper limit
	bounds = list(zip(ll*np.ones(pol0.size),up*np.ones(pol0.size))) # Bind values
	return scipy.optimize.minimize(objective_function, pol0,
							bounds=bounds, 
							args=(pol0, des, alpha, H, b0, A, E))
 
def main(pol0, des, H, A, E):
	#### Calculate estimated alpha using ratio of H to E for each row ####
	# Solve for alpha as follows
	# Use: r = H/E = alpha/(1-alpha) (based on alpha = probability of H, (1-alpha) = probability of E)
	# Then: solve for alpha --> alpha = H/E / (1+H/E)
	with np.errstate(divide='ignore',invalid='ignore'):
		r = H.sum(axis=1) / E.sum(axis=1)
	r = np.nan_to_num(r) # Remove NaN Just in case
	alpha = r / (1 + r)

	# Normalize
	for i in range(0,A.shape[0]): A[i] = matop.normalize_rows(A[i])
	H = matop.normalize_rows(H)
	E = matop.normalize_rows(E)

	# Get b0
	b0 = update_b(A, pol0) # b0 holds the total # of transitions weighted by the policy pol0

	#### Optimize using pagerank fitness ####
	result = optimize(pol0, des, alpha, H, b0, A, E)

	#### Extract output ####
	policy = result.x
	cols = pol0.shape[1] # Number of columns (as per input policy)
	policy = np.reshape(policy,(policy.size//cols,cols)) # Resize pol
	if cols > 1: policy = matop.normalize_rows(policy) # Normalize rows

	#### Extract unknown states (for analysis purposes) ####
	temp = H + E # All transitions
	empty_cols = np.where(~temp.any(axis=0))[0]
	empty_rows = np.where(~temp.any(axis=1))[0]
	empty_states = np.intersect1d(empty_cols,empty_rows,assume_unique=True)

	return result, policy, empty_states
