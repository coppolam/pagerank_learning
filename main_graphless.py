import numpy as np
import fileHandler as fh
import matrixOperations as matOp
import scipy.optimize as spopt

import os
import subprocess

pol_sim = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5])
pol_test = np.array([0,0,0,0,0,0,0])
des = np.array([2,3,4,5,6])
alpha = 0.55

verbose = 1

def fitness(pr,des):
	return np.mean(pr[:,des])/np.mean(pr)

def objective_function(pol,H,E,A):
	pol = np.around(pol)
	Hnew = matOp.update_H(H, A, E, pol_sim, pol)
	G = alpha * Hnew + (1 - alpha) * E
	pr = matOp.pagerank(G)
	f = fitness(pr,des)
	if verbose > 1:
		print("Fitness \tf = " + str(round(f,5))+ "\t1/f = " + str(round(1/f,5)))
	return 1 / f # Trick it into maximizing

def optimize(H,E,A):
	s = "randtobest1bin"
	bounds = list(zip(np.zeros(np.size(pol_sim)),np.ones(np.size(pol_sim)))) # Bind values
	result = spopt.differential_evolution(objective_function,
                                       bounds=bounds, 
                                       args=(H,E,A),
									   strategy=s)
	
	if verbose > 0:
		print("\n*****Result******")
		print(result)
		print("Policy: " + str(np.around(result.x)))

	return result.x, result.fun

def test():
	objective_function(pol_test)
 
def main():
	# Launch Swarmulator
	i = 0

	while i < 10:
		i = i + 1
		
		# Empty files
		tmp = subprocess.call("cd ../swarmulator/mat/ && rm *.csv", shell=True)
  		  
		# Call the simulator
		tmp = subprocess.call("cd ../swarmulator && ./swarmulator 20", shell=True)
  		
		# Read output
		H, A, E = fh.read_matrices("../swarmulator/mat/")
		H = matOp.normalize_rows(H);
		E = matOp.normalize_rows(E);
		# E = matOp.normalize_rows(matOp.make_binary(E))
		policy, fitness = optimize(H,E,A)
  
if __name__ == '__main__':
	main()