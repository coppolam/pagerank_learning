import numpy as np

def normalize_rows(mat):
	row_sums = np.sum(mat, axis=1)
	mat = np.divide(mat,row_sums[:,np.newaxis], 
		out=np.zeros_like(mat), where=row_sums[:,np.newaxis]!=0)
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

def update_H(H, A ,E , pol_sim, pol):
	# Update H based on actions
	b = A * pol_sim[:, np.newaxis]
	Hnew = np.divide(H, b, out=np.zeros_like(H), where=b!=0);
	Hnew = Hnew * ( A * pol[:, np.newaxis]);
	return Hnew
