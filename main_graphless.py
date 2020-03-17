import numpy as np
import fileHandler as fh
import matrixOperations as matOp
import scipy.optimize as spopt

import os
import subprocess
import networkx as nx

pol_sim = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5])
pol_test = np.array([0,0,0,0,0,0,0])
des = np.array([2,3,4,5,6])

verbose = 1

def fitness(pr,des):
	return np.mean(pr[:,des])/np.mean(pr)

def objective_function(pol,alpha,H,E,A):
	# pol = np.around(pol)
	Hnew = matOp.update_H(H, A, E, pol_sim, pol)
	ad = np.diag(alpha)
	ad1 = np.diag(1-alpha)
	G = ad.dot(Hnew) + ad1.dot(E)
	# G = alpha * Hnew + (1-alpha) * E
	G = matOp.normalize_rows(G)
	gr = nx.from_numpy_matrix(Hnew)
	
	if not nx.is_connected(gr):
		f = 0
	else:
		pr = matOp.pagerank(G)
		a = np.isinf(pr)
		f = fitness(pr,des)
	if verbose > 1:
		print("Fitness \tf = " + str(round(f,5))+ "\t1/(1+f) = " + str(round(1/(1+f),5)))
	return 1 / (f+1) # Trick it into maximizing

def optimize(alpha,H, E, A):
	s = "randtobest1bin"
	# s = "best2bin"
	bounds = list(zip(np.zeros(np.size(pol_sim)),np.ones(np.size(pol_sim)))) # Bind values
	result = spopt.differential_evolution(objective_function,
                                       bounds=bounds, 
                                       args=(alpha,H,E,A),
									   strategy=s,
									   popsize=30,
									   maxiter=2000)
	
	if verbose > 0:
		print("\n*****Result******")
		print(result)
		print("Policy: " + str(result.x))

	return result.x, result.fun

def test():
	objective_function(pol_test)
 
def launch_sim():
    # Launch Swarmulator
	subprocess.call("cd ../swarmulator/mat/ && rm *.csv", shell=True)
	subprocess.call("cd ../swarmulator && ./swarmulator 20", shell=True)
	
def main():
	i = 1

	# Empty files and call swarmulator
	try:
		# launch_sim()
		print("Sim")
	except:
		raise ValueError("Could not launch the simulator!")

	while i < 2:
		print("\n***********************************")
		print("Run "+str(i))
		print("***********************************")
		i = i + 1
		
		# Read output
		H, A, E = fh.read_matrices("../swarmulator/mat/")
		np.set_printoptions(suppress=True)
		print(H)
		print(E)
		
		alpha = np.sum(H,axis=1)/np.sum(E,axis=1)
		alpha = np.nan_to_num(alpha)
		# alpha = 0.5
		# print(alpha)
		alpha = alpha / (1+alpha)
		print(alpha)
		# H = matOp.normalize_rows(H);
		# E = matOp.normalize_rows(E);
		# print(H)
		# E = matOp.normalize_rows(matOp.make_binary(E))
		policy, fitness = optimize(alpha,H,E,A)

if __name__ == '__main__':
	main()