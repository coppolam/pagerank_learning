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

verbose = 1 # 0 barely, 1 = some, 2 = a lot

def update_H(H, A ,E , pol0, pol):
    # Update H based on actions

	# Iterate over actions (columns of pol0)
	b0 = np.zeros(A.shape)
	i = 0
	for p in pol0.T:
		i += 1
		Atemp = np.zeros(A.shape)
		Atemp[np.where(A==int(i))] = 1
		b0 += Atemp * p[:, np.newaxis]
	
	# Iterate over new actions (columns of pol)
	cols = pol0.shape[1]
	pol = np.reshape(pol,(pol.size//cols,cols))# Resize pol
	if cols > 1: 
		pol = matop.normalize_rows(pol)
	b1 = np.zeros(A.shape)
	i = 0
	for p in pol.T:
		i += 1
		Atemp = np.zeros(A.shape)
		Atemp[np.where(A==int(i))] = 1
		b1 += Atemp * p[:, np.newaxis]

	Hnew = np.divide(H, b0, out=np.zeros_like(H), where=b0!=0);
	Hnew = Hnew * b1;

	return Hnew

def fitness(pr,des):
    return np.average(pr,axis=1,weights=des)/pr.mean()

def objective_function(pol, pol0, des, alpha, H, A, E):
	Hnew = update_H(H, A, E, pol0, pol)
	G = np.diag(alpha).dot(Hnew) + np.diag(1-alpha).dot(E)
	pr = matop.pagerank(G)
	f = fitness(pr, des)
	if verbose > 1:
		print(str(round(pol[0],5)) + 
			" Fitness \tf = " + str(round(f,5)) + 
			"\t1/(1+f) = " + str(round(1/(f + 1),5)))
	return 100 / (f + 1) # Trick it into maximizing

def optimize(pol0, des, alpha, H, A, E):
	# Bound probabilistic policy
	ll = 0. # Lower limit
	up = 1. # Upper limit
	
	# Optimize
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

	policy = result.x
	cols = pol0.shape[1]
	policy = np.reshape(policy,(policy.size//cols,cols)) # Resize pol
	# if cols > 1:
	# 	policy = matop.normalize_rows(policy)

	return result, policy, empty_states

if __name__ == '__main__':
	main()