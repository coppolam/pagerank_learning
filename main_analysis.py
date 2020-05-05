import aggregation as env
import pickle

# Load environment
sim = env.aggregation()
	
run = False
r = [10,20,30,40,50]
tmax = 10000
if run:
	file = "data/1_learning_data_t%i_r%i.npz"
	from tools import fitness_functions as ff

	# Load all and re-evaluate global and local fitnesses
	counter = 0
	data = {"t":[], "f":[], "s":[]}
	for c in r:
		sim.load(file=(file %(tmax,c)))
		sim.sim.plot_log(file=(file %(tmax,c)))
		t, f, s = sim.reevaluate(ff.mean_number_of_neighbors, ff.mean_number_of_neighbors)
		data["t"].append(t)
		data["f"].append(f)
		data["s"].append(s)
		counter += 1

	# Save
	with open("data.pkl", "wb") as cp_file:
		pickle.dump(data, cp_file)

# Load
with open("data.pkl", "rb") as cp_file:
	data = pickle.load(cp_file)
			
# Compare
import matplotlib.pyplot as plt
import numpy as np
symbols = ["*",".","+","."]
counter = len(data["f"])
for c in range(0,counter):
	plt.plot(data["f"][c][:,0]*r[c],data["f"][c][:,1],symbols[c])
	corr = np.corrcoef(data["f"][c][:,0]*r[c],data["f"][c][:,1])[0,1]
	print(corr)
plt.show()
