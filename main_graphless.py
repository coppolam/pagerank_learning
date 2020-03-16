import numpy as np
import fileHandler as fh
import matrixOperations as matOp
import scipy.optimize as spopt

pol_sim = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5])
pol_test = np.array([0,0,0,0,0,0,0])
des = np.array([2,3,4,5,6])
alpha = 0.55

H, A, E = fh.read_matrices()
H = matOp.normalize_rows(H);
E = matOp.normalize_rows(E);
# E = matOp.normalize_rows(matOp.make_binary(E))

def fitness(pr,des):
	return np.mean(pr[:,des])/np.mean(pr)

def objective_function(pol):
	pol = np.around(pol)
	Hnew = matOp.update_H(H, A, E, pol_sim, pol)
	G = alpha * Hnew + (1 - alpha) * E
	pr = matOp.pagerank(G)
	f = fitness(pr,des)
	print("Fitness \tf = " + str(round(f,5))+ "\t1/f = " + str(round(1/f,5)))
	return 1 / f # Trick it into maximizing

def optimize():
	s = "randtobest1bin"
	bounds = list(zip(np.zeros(np.size(pol_sim)),np.ones(np.size(pol_sim)))) # Bind values
	result = spopt.differential_evolution(objective_function, bounds,
                                       strategy=s)
	
	print("\n*****Result******")
	print(result)
	print(np.around(result.x))

	return result.x, result.fun

def test():
	objective_function(pol_test)
 
def main():
    # test()
	policy, fitness = optimize()
	print(np.around(policy))

if __name__ == '__main__':
	main()