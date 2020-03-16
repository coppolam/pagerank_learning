import numpy as np

def normalize_rows(mat):
	row_sums = np.sum(mat, axis=1)
	mat = np.divide(mat,row_sums[:,np.newaxis], 
				 out=np.zeros_like(mat), 
				 where=row_sums[:,np.newaxis]!=0)
	return mat

def pagerank(G, tol=1e-8):
	# Iterative procedure
	n = G.shape[0] # Size of G
	pr = 1 / n * np.ones((1, n)) # Initialize Pagerank vector
	residual = 1 # Residual (initialize high, doesn't really matter)
	while residual >= tol:
		pr_previous = pr
		pr = np.matmul(pr,G) # Pagerank formula
		residual = np.linalg.norm(np.subtract(pr,pr_previous))
	return normalize_rows(np.asarray(pr))

def update_H(H, A ,E , pol_sim, pol):
	# Update H based on actions
	b = A * pol_sim[:, np.newaxis]
	Hnew = np.divide(H, b, out=np.zeros_like(H), where=b!=0);
	Hnew = Hnew*(A*pol[:, np.newaxis]);
	Hnew = normalize_rows(Hnew)
	
	# Add static rows from E ("D" matrix)
	static_rows = np.where(~Hnew.any(axis=1))[0]
	Hnew[static_rows,:] = E[static_rows,:]
	return Hnew

def make_binary(mat):
	mat[mat > 0] = 1
	return mat
