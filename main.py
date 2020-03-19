import consensus
import graphless_optimization
import numpy as np

f, policy, perms, des, H, A, E = consensus.run(1)
print(f)
print(H)
print(A)
print(E)
print(des)
print(perms)
result = graphless_optimization.main(policy, des, H, A, E)

pol = result.x
cols = np.size(policy,1)
pol = np.reshape(pol,(np.size(pol)//cols,cols))# Resize pol
print(pol)