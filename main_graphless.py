import numpy as np
import fileHandler as fh
import matrixOperations as matop
import scipy.optimize as spopt
import subprocess

np.set_printoptions(suppress=True) # Avoid scientific notation
		
pol_sim = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5])
des = np.array([2,3,4,5,6])
verbose = 1 # 0 barely, 1 = some, 2 = a lot
runs = 1

def fitness(pr,des):
	return np.mean(pr[:,des])/np.mean(pr)

def objective_function(pol, alpha, H, E, A):
	Hnew = matop.update_H(H, A, E, pol_sim, pol)
	G = np.diag(alpha).dot(Hnew) + np.diag(1-alpha).dot(E)
	pr = matop.pagerank(G)
	f = fitness(pr,des)
	if verbose > 1:
		print(str(round(pol[0],5)) + 
			" Fitness \tf = " + str(round(f,5)) + 
			"\t1/(1+f) = " + str(round(1/(f + 1),5)))
	return 100 / (f + 1) # Trick it into maximizing

def optimize(alpha, H, E, A):
	# Bound probabilistic policy to being between 0 and 1
	bounds = list(zip(np.zeros(np.size(pol_sim)),np.ones(np.size(pol_sim)))) # Bind values
	
	# Optimize
	result = spopt.minimize(objective_function,pol_sim,
                                       bounds=bounds, 
                                       args=(alpha,H,E,A))
	
	return result
 
def launch_simulator(i):
	subprocess.call("cd ../swarmulator/mat/ && mkdir hist" + str(i) + " && mv *.csv hist" + str(i) + "/", shell=True)
	subprocess.call("cd ../swarmulator && ./swarmulator 20", shell=True)
	
def main():
    	
	i = 0
	while i < runs:
		launch_simulator(i)
		
		# Read output of simulation
		H, A, E = fh.read_matrices("../swarmulator/mat/")
		r = np.sum(H, axis=1) / np.sum(E, axis=1)
		r = np.nan_to_num(r) # Just in case
		alpha = r / (1 + r)
		
		result = optimize(alpha, H, E, A)

		if verbose > 0:
			print("\n*********** RUN " + str(i + 1) + " ***********")
			print(H), print(E), print(alpha)
			print("\n*****Result******")
			print("Fitness: " + str(result.fun))
			print("Policy: " + str(result.x))

		i = i + 1

if __name__ == '__main__':
	main()