#!/usr/bin/env python3
"""
Optimize a behavior based on the PageRank function
@author: Mario Coppola, 2020
"""

import numpy as np
from tools import fileHandler as fh
from tools import matrixOperations as matop
import scipy.optimize as spopt
np.set_printoptions(suppress=True) # Avoid scientific notation
c = 0
verbose = 2 # 0 barely, 1 = some, 2 = a lot
import sys


def update_b(A,pol):
	b = np.zeros(A.shape)
	i = 0
	for p in pol.T:  # Iterate over each action (columns of pol0)
		i += 1 # Start with action i=1 (action i=0 is reserved for E matrix)
		# temp = np.zeros(A.shape) # Initialize temporary matrix of size H with zeros
		# temp[np.where(A==int(i))] = 1 # Set to 1 if the action is responsible for state transition
		b += A * p[:, np.newaxis] # Multiply by the policy
	return b

def update_H(H, A ,E , pol0, pol):
    	
	# Reshape policy vector to the right matrix dimensions and normalize
	cols = pol0.shape[1]
	pol = np.reshape(pol,(pol.size//cols,cols)) # Resize pol
	if cols > 1: pol = matop.normalize_rows(pol+0.001) # 0.001 as nonzero hack

    ### Routine to update H based on new policy ###

	### Step 1. Remove the known impact from the sim
	b0 = update_b(A,pol0)
	# Remove b0 from H
	# (unless no action is associated in which case we just get a row of zeros
	Hnew = np.divide(H, b0, out=np.zeros_like(H), where=b0!=0);
	
	### Step 2. Reassign to new policy
	b = update_b(A,pol)
	Hnew = Hnew * b;

	return Hnew, pol

def fitness(pr,des):
    return np.average(pr,axis=1,weights=des)/pr.mean()

def objective_function(pol, pol0, des, alpha, H, A, E):
	global c
	Hnew, pol = update_H(H, A, E, pol0, pol) # Update H with new policy
	# psum = np.sum(pol0, axis=1)
	G = np.diag(alpha).dot(Hnew) + np.diag(1-alpha).dot(E) # Google formula
	pr = matop.pagerank(G) # Evaluate pagerank vector 
	f = fitness(pr, des) # Get fitness
	if verbose > 1:
		sys.stdout.write("\r Fitness \tf = " + str(np.round(f,5)) + "\t1/(1+f) = " + str(np.round(1/(f + 1),5)))
		sys.stdout.flush()
		# print(pol)
		# print("\r Fitness \tf = " + str(np.round(f,10)) + 
		# 	"\t1/(1+f) = " + str(np.round(1/(f + 1),10)))
		
	return 1 / (f + 1) # Trick it into maximizing

def optimize(pol0, des, alpha, H, A, E):
	# Bound probabilistic policy
	ll = 0.  # Lower limit
	up = 1.0 # Upper limit
	bounds = list(zip(ll*np.ones(pol0.size),up*np.ones(pol0.size))) # Bind values
	result = spopt.minimize(objective_function, pol0,
										bounds=bounds,
										args=(pol0, des, alpha, H, A, E))

	return result
 
def main(pol0, des, H, A, E):
	# Find unknown states
	temp = H + E
	empty_cols = np.where(~temp.any(axis=0))[0]
	empty_rows = np.where(~temp.any(axis=1))[0]
	empty_states = np.intersect1d(empty_cols,empty_rows,assume_unique=True)
	# Although this ok in principle, it would be better to do it
	# at simulation time, rather than to make a big H and then
	# proceed to cut it.
	# H = np.delete(H, empty_states, empty_states)
	# A = np.delete(A, empty_states, empty_states)
	# E = np.delete(E, empty_states, empty_states)
	
	with np.errstate(divide='ignore',invalid='ignore'):
		r = H.sum(axis=1) / E.sum(axis=1)
	r = np.nan_to_num(r) # Just in case
	alpha = r / (1 + r)
	
	result = optimize(pol0, des, alpha, H, A.astype("int"), E)
	print("\nDone")
	policy = result.x
	cols = pol0.shape[1]
	policy = np.reshape(policy,(policy.size//cols,cols)) # Resize pol
	if cols > 1:
		policy = matop.normalize_rows(policy)
	print(alpha)
	return result, policy, empty_states
