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

verbose = 2

def fitness(pr,des):
	return np.mean(pr[:,des])/np.mean(pr)

def objective_function(pol,alpha,H,E,A):
	Hnew = matOp.update_H(H, A, E, pol_sim, pol)
	G = np.diag(alpha).dot(Hnew) + np.diag(1-alpha).dot(E)
	pr = matOp.pagerank(G)
	f = fitness(pr,des)
	if verbose > 1:
		print(str(round(pol[0],4)) + 
			" Fitness \tf = " + str(round(f,5)) + 
			"\t1/(1+f) = " + str(round(1/(f+1),5)))
	return 1 / (f+1) # Trick it into maximizing

def optimize(alpha,H, E, A):
	s = "randtobest1bin"
	# s = "best2bin"
	# s = "best2exp"

	# Answers are bound between 0 and 1
	bounds = list(zip(np.zeros(np.size(pol_sim)),np.ones(np.size(pol_sim)))) # Bind values
	
	# Optimize
	result = spopt.differential_evolution(objective_function,
                                       bounds=bounds, 
                                       args=(alpha,H,E,A),
									   strategy=s,
									   popsize=15,
									   maxiter=1000)
	
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
		
		# Read output of simulation
		H, A, E = fh.read_matrices("../swarmulator/mat/")
		np.set_printoptions(suppress=True)
		r = np.sum(H, axis=1) / np.sum(E, axis=1)
		r = np.nan_to_num(r) # Just in case
		alpha = r / (1 + r)
		
		if verbose > 1:
			print(H)
			print(E)
			print(alpha)

		policy, fitness = optimize(alpha,H,E,A)

if __name__ == '__main__':
	main()