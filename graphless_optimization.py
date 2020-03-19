import numpy as np
from tools import fileHandler as fh
from tools import matrixOperations as matop
import scipy.optimize as spopt
np.set_printoptions(suppress=True) # Avoid scientific notation
		
verbose = 1 # 0 barely, 1 = some, 2 = a lot

def update_H(H, A ,E , pol_sim, pol):
    # Update H based on actions

	# Iterate over actions (columns of pol_sim)
	b0 = np.zeros([np.size(A,0),np.size(A,1)])
	i = 0
	for p in pol_sim.T:
		i += 1
		Atemp = np.zeros([np.size(A,0),np.size(A,1)])
		Atemp[np.where(A==int(i))] = 1
		b0 += Atemp * p[:, np.newaxis]
	
	# Iterate over new actions (columns of pol)
	cols = np.size(pol_sim,1)
	pol = np.reshape(pol,(np.size(pol)//cols,cols))# Resize pol
	b1 = np.zeros([np.size(A,0),np.size(A,1)])
	i = 0
	for p in pol.T:
		i += 1
		Atemp = np.zeros([np.size(A,0),np.size(A,1)])
		Atemp[np.where(A==int(i))] = 1
		b1 += Atemp * p[:, np.newaxis]
	
	Hnew = np.divide(H, b0, out=np.zeros_like(H), where=b0!=0);
	Hnew = Hnew * b1;
	return Hnew

def fitness(pr,des):
	return np.mean(pr[:,des])/np.mean(pr)

def objective_function(pol, pol_sim, des, alpha, H, A, E):
	Hnew = update_H(H, A, E, pol_sim, pol)
	G = np.diag(alpha).dot(Hnew) + np.diag(1-alpha).dot(E)
	pr = matop.pagerank(G)
	f = fitness(pr, des)
	if verbose > 1:
		print(str(round(pol[0],5)) + 
			" Fitness \tf = " + str(round(f,5)) + 
			"\t1/(1+f) = " + str(round(1/(f + 1),5)))
	return 100 / (f + 1) # Trick it into maximizing

def optimize(pol_sim, des, alpha, H, A, E):
	# Bound probabilistic policy to being between 0 and 1
	bounds = list(zip(np.zeros(np.size(pol_sim)),np.ones(np.size(pol_sim)))) # Bind values
	
	# Optimize
	result = spopt.minimize(objective_function, pol_sim,
                                       bounds=bounds, 
                                       args=(pol_sim, des, alpha, H, A, E))
	
	return result
 
# def launch_simulator(i):
# 	subprocess.call("cd ../swarmulator/mat/ && mkdir hist" + str(i) + " && mv *.csv hist" + str(i) + "/", shell=True)
# 	subprocess.call("cd ../swarmulator && ./swarmulator 20", shell=True)
	
def main(pol_sim, des, H, A, E):
	# i = 0
	# while i < runs:
	# launch_simulator(i)
	
	# Read output of simulation
	# H, A, E = fh.read_matrices("../swarmulator/mat/")
	r = np.sum(H, axis=1) / np.sum(E, axis=1)
	r = np.nan_to_num(r) # Just in case
	alpha = r / (1 + r)
	
	result = optimize(pol_sim, des, alpha, H, A.astype("int"), E)

	if verbose > 0:
		print("\n*****Result******")
		print("Fitness: " + str(result.fun))
		print("Policy: " + str(result.x))

	# i = i + 1

	return result

if __name__ == '__main__':
	main()