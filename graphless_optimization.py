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
	i = 0 # Itarator
	for p in pol.T: # Iterate over each action (columns of pol0)
		temp = A[i].astype(float)
		b += np.divide(temp * p, H, out=np.zeros_like(H), where=H!=0) # Multiply by the policy
		i += 1
		
	return b

def update_H(H0, A ,E , pol0, pol):	
	# Reshape policy vector to the right matrix dimensions and normalize
	cols = pol0.shape[1]
	pol = np.reshape(pol,(pol.size//cols,cols)) # Resize pol
	if cols > 1: pol = matop.normalize_rows(pol)

    ###########################################
	# Routine to update H based on new policy #
	###########################################

	### Step 1. Remove the known impact from the sim
	# TODO: Step 1 is always the same, move outside to speed up
	b0 = update_b(H0,A,pol0)
	# Remove b0 from H (unless no action is associated 
	# in which case we just get a row of zeros)
	Ht = np.divide(H0, b0, out=np.zeros_like(H0), where=b0!=0);
	
	### Step 2. Reassign to new policy
	b1 = update_b(H0,A,pol)
	H1 = Ht * b1;

	return H1, pol

def fitness(pr,des):
    return np.average(pr,axis=1,weights=des)/pr.mean()

def objective_function(pol, pol0, des, alpha, H, A, E):
	Hnew, pol = update_H(H, A, E, pol0, pol) # Update H with new policy
	# psum = np.sum(pol, axis=1)
	G = np.diag(alpha).dot(Hnew) + np.diag(1-alpha).dot(E) # Google formula
	pr = matop.pagerank(G) # Evaluate pagerank vector 
	f = fitness(pr, des) # Get fitness
	# Display to terminal
	sys.stdout.write("\r Fitness \t maximizing f = " + str(np.round(f,5)) + "\t, minimizing 1/(1+f) = " + str(np.round(1/(f + 1),5)))
	sys.stdout.flush()

	return 1 / (f + 1) # Trick scipy into maximizing

def optimize(pol0, des, alpha, H, A, E):
	# Bind probabilistic policy
	ll = 0.001  # Lower limit 0.001 because ll!=0
	up = 1.0 # Upper limit
	bounds = list(zip(ll*np.ones(pol0.size),up*np.ones(pol0.size))) # Bind values
	result = scipy.optimize.minimize(objective_function, pol0,
							bounds=bounds, 
							args=(pol0, des, alpha, H, A, E))
	return result
 
def main(pol0, des, H, A, E):
    #### Calculate estimated alpha using ratio of H to E for each row ####
	# Solve for alpha as follows
	# Use: r = H/E = alpha/(1-alpha) (based on alpha = probability of H, (1-alpha) = probability of E)
	# Then: solve for alpha --> alpha = H/E / (1+H/E)
	with np.errstate(divide='ignore',invalid='ignore'):
		r = H.sum(axis=1) / E.sum(axis=1)
	r = np.nan_to_num(r) # Remove NaN Just in case
	alpha = r / (1 + r)
	
	#### Optimize using pagerank fitness ####
	result = optimize(pol0, des, alpha, H, A.astype("int"), E)
	print("\nDone")

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
