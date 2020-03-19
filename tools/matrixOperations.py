#!/usr/bin/env python3
"""
Collection of tools for "less standard" math operations
@author: Mario Coppola, 2020
"""
import numpy as np

def round_to_multiple(a, mult):
    return np.around(a / mult) * mult

def normalize_rows(mat,axis=1):
	row_sums = np.sum(mat, axis=axis)
	if not np.isscalar(row_sums):
		mat = np.divide(mat,row_sums[:,np.newaxis], out=np.zeros_like(mat), where=row_sums[:,np.newaxis]!=0)
	else:
		mat = np.divide(mat,row_sums)
	return mat

def pagerank(G, tol=1e-8):
	# Iterative procedure to solve for the PageRank vector
	G = normalize_rows(G)
	n = G.shape[0]
	pr = 1 / n * np.ones((1, n)) # Initialize PageRank vector
	residual = 1 # Initialize residual
	
	while residual >= tol:
		pr_previous = pr
		pr = np.matmul(pr,G) # Pagerank formula
		residual = np.linalg.norm(np.subtract(pr,pr_previous))
	
	return normalize_rows(np.asarray(pr))
