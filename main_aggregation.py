
import graphless_optimization as opt
from tools import fileHandler as fh
from tools import matrixOperations as matop
import numpy as np
import subprocess

###### Simulate ######
from simulator import swarmulator
rerun = False
if rerun:
    subprocess.call("cd ../swarmulator/mat/ && rm *.csv", shell=True)
    s = swarmulator.swarmulator("../swarmulator") # Init
    s.make(clean=False,animation=True) # Build (if already built, you can skip this)
    f = s.run(20) # Run it, and receive the fitness.

###### Optimize ######
folder = "../swarmulator/mat/"
H = fh.read_matrix(folder,"H")
A = fh.read_matrix(folder,"A")
E = fh.read_matrix(folder,"E")
policy = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5])
des = np.array([2,3,4,5,6])
policy = policy.reshape(np.size(policy),1)
result, policy, empty_states = opt.main(policy, des, H, A, E)
fh.save_data(folder + "optimization", des, result, policy)

print('{:=^40}'.format(' Optimization '))
print("Final fitness: " + str(result.fun))
print("[ policy ]")
print(policy)
