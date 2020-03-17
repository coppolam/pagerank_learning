import numpy as np
import fileHandler as fh
import matrixOperations as matOp
import scipy.optimize as spopt
import os
import subprocess

np.set_printoptions(suppress=True) # Avoid scientific notation
		
pol_sim = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5])
des = np.array([2,3,4,5,6])
verbose = 1 #0 barely, 1 = some, 2 = a lot

def fitness(pr,des):
	return np.mean(pr[:,des])/np.mean(pr)

def objective_function(pol, alpha, H, E, A):
	Hnew = matOp.update_H(H, A, E, pol_sim, pol) # Update H
	G = np.diag(alpha).dot(Hnew) + np.diag(1-alpha).dot(E)
	pr = matOp.pagerank(G)
	f = fitness(pr,des)
	if verbose > 1:
		print(str(round(pol[0],5)) + 
			" Fitness \tf = " + str(round(f,5)) + 
			"\t1/(1+f) = " + str(round(1/(f + 1),5)))
	return 1 / (f + 1) # Trick it into maximizing

def optimize(alpha, H, E, A):
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
									   popsize=30,
									   maxiter=2000)
	
	if verbose > 0:
		print("\n*****Result******")
		print(result)
		print("Policy: " + str(result.x))

	return result.x, result.fun
 
def launch_simulator():
    # Launch Swarmulator
	subprocess.call("cd ../swarmulator/mat/ && rm *.csv", shell=True)
	subprocess.call("cd ../swarmulator && ./swarmulator 20", shell=True)
	
def main():
	runs = 1
	i = 0
	while i < runs:
		launch_simulator()

		print("\n***********************************")
		print("Run " + str(i + 1))
		print("***********************************")
		i = i + 1
		
		# Read output of simulation
		H, A, E = fh.read_matrices("../swarmulator/mat/")
		r = np.sum(H, axis=1) / np.sum(E, axis=1)
		r = np.nan_to_num(r) # Just in case
		alpha = r / (1 + r)
		
		if verbose > 0:
			print(H)
			print(E)
			print(alpha)

		policy, fitness = optimize(alpha, H, E, A)

if __name__ == '__main__':
	main()