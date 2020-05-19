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

def update_b(H,A,pol):
	b = np.zeros(H.shape)
	i = 0 # Iterator
	cols = pol.shape[0]
	for p in pol.T: # Iterate over each action (columns of pol0)
		if cols == 1: Ap = A[i].astype(float) * p[:, np.newaxis]
		else: Ap = A[i].astype(float) * p[:, np.newaxis]
		b += np.divide(Ap, H, out=np.zeros_like(H), where=H!=0) # Multiply by the policy
		i += 1
	# Matrix b holds the probability of the transition happening weighted by the action, 
	# which is the probability of the action times the relative probability 
	# of the action leading to a particular state change.
	return b

def update_H(H0, Ht, A ,E , pol0, pol):	
	# Reshape policy vector to the right matrix dimensions and normalize
	cols = pol0.shape[1]
	pol = np.reshape(pol,(pol.size//cols,cols)) # Resize policy
	if cols > 1: pol = matop.normalize_rows(pol+0.001) # Normalize +0.001 to keep connected

    ###########################################
	# Routine to update H based on new policy #
	###########################################
	# We first strip H of the impact of the policy, then reassign it with the new policy
	### Step 1. Remove the known impact from the sim
	# We do this once in the beginning since it's always the same operation

	### Step 2. Reassign to new policy
	b1 = update_b(H0, A, pol) # Use H0 since that holds the total # of transitions, we only change the weighing
	return Ht * b1;

def fitness(pr,des):
    return np.average(pr,axis=1,weights=des)/pr.mean()

def objective_function(pol, pol0, des, alpha, H, Ht, A, E):
	H1 = update_H(H, Ht, A, E, pol0, pol) # Update H with new policy
	G = np.diag(alpha).dot(H1) + np.diag(1-alpha).dot(E) # Google matrix
	pr = matop.pagerank(G) # Evaluate pagerank vector 
	f = fitness(pr, des) # Get fitness
	
	# Display to terminal
	p = "\r Fitness \t max:f=%.10f \t min:1/(1+f)=%.10f" % (f, 1/(f+1))
	sys.stdout.write(p)
	sys.stdout.flush()

	return 1 / (f + 1) # Trick scipy into maximizing

def optimize(pol0, des, alpha, H, Ht, A, E):
	# Bind probabilistic policy
	ll = 0.0 # Lower limit
	up = 1.0 # Upper limit
	bounds = list(zip(ll*np.ones(pol0.size),up*np.ones(pol0.size))) # Bind values
	return scipy.optimize.minimize(objective_function, pol0,
							bounds=bounds, 
							args=(pol0, des, alpha, H, Ht, A, E))
 
def main(pol0, des, H, A, E):
	#### Calculate estimated alpha using ratio of H to E for each row ####
	# Solve for alpha as follows
	# Use: r = H/E = alpha/(1-alpha) (based on alpha = probability of H, (1-alpha) = probability of E)
	# Then: solve for alpha --> alpha = H/E / (1+H/E)
	with np.errstate(divide='ignore',invalid='ignore'):
		r = H.sum(axis=1) / E.sum(axis=1)
	r = np.nan_to_num(r) # Remove NaN Just in case
	alpha = r / (1 + r)

	### Step 1. Remove the known impact from the sim
	b0 = update_b(H, A, pol0) # Use H0 which holds the total # of transitions, we only change the weighing
	# Remove b0 from H (unless no action is associated 
	# in which case we just get a row of zeros)
	Ht = np.divide(H, b0, out=np.zeros_like(H), where=b0!=0);
	
	#### Optimize using pagerank fitness ####
	result = optimize(pol0, des, alpha, H, Ht, A, E)

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
