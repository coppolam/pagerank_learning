#!/usr/bin/env python3
"""
Collection of tools for "less standard" and/or useful 
math and matrix operations

@author: Mario Coppola, 2020
"""

import numpy as np

def round_to_multiple(a, mult, floor=True):
	'''Rounds a number to its closest multiple'''
	
	# If floor=True, floor the values, otherwise round them
	if floor:
		a = np.floor(a / mult)
	else:
		a = np.round(a / mult)

	# Return the rounded one
	return a * mult

def normalize_rows(mat):
	'''Normalizies the rows of a matrix'''

	# Get the sum of the rows
	## If the input is 2D, then we take row wise sums
	## Otherwise we sum all elements
	if isinstance(mat,list):
		if isinstance(mat[0],list):
			axis = 1
		else:
			axis = 0
	elif isinstance(mat,np.ndarray):
		if mat.ndim == 2:
			axis = 1
		else:
			axis = 0
	else:
		print("Warning: normalize_rows unrecognized type. \
				Expected list or numpy array.")
		axis = 1

	row_sums = np.sum(mat, axis=axis)

	# If any of the matrices has elements larger than 0
	if np.any(row_sums>0):

		# If the row_sums is not a scalar then we want to be a bit fancier
		if not np.isscalar(row_sums):
			# Normalize (but ignore a row that is entirely 0s)
			mat = np.divide(mat,row_sums[:,np.newaxis], 
				out=np.zeros_like(mat),
				where=row_sums[:,np.newaxis]!=0)

		else:
			mat = np.divide(mat,row_sums)

	return mat

def pagerank(G, pr=None, tol=1e-8, maxiter=5000):
	'''Iterative procedure to solve for the PageRank vector'''

	# Normalize the rows of the Google Matrix
	G = normalize_rows(G)

	# If the pagerank vector is not specified, initialize it
	if pr is None:
		n = G.shape[0]    		
		pr = 1 / n * np.ones((1, n))

	# Initialize residual
	residual = 1

	# Run the PageRank iteration for a maximum number of iterations
	# or until the residual is smaller than the tolerance value tol

	i = 0
	while residual >= tol and i < maxiter:
		# Update pagerank vector
		pr_previous = pr

		# Pagerank formula
		pr = np.matmul(pr,G)

		# Calculate residual
		residual = np.linalg.norm(np.subtract(pr,pr_previous))
		i += 1
	
	# Return the normalized pagerank vector
	return normalize_rows(np.asarray(pr))

def pretty_print(mat):
	'''Prints a matrix to the terminal but it looks a little better'''
	for x in mat:
		print(*x, sep=" ")
