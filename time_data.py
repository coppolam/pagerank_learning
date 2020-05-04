import numpy as np
from simulator import swarmulator
import time

sim = swarmulator.swarmulator(verbose=True)
	
def wall_clock_test(n,m):
	t = np.zeros([n,m])
	for i in range(1,n+1): # Itarate over number of robots
		for j in range(0,m): # Do multiple runs to average out
			tic = time.time()    
			f = sim.run(i) # Launch simulation run
			toc = time.time()
			t[i-1,j] = toc-tic
			print("New result %i/%i, %i/%i: %.2f" % (i,n,j+1,m,t[i-1,j]))
	return t

def wall_clock_batch_test(n,m,batch):
	t = np.zeros([n,m])
	for i in range(1,n+1): # Itarate over number of robots
		for j in range(0,m): # Do multiple runs to average out
			tic = time.time()
			f = sim.batch_run(i,batch) # Launch batch
			toc = time.time()
			t[i-1,j] = (toc-tic)
			print("New result %i/%i, %i/%i: %.2f" % (i,n,j+1,m,t[i-1,j]))
	return t

def run(n,m,batch,filename=None):
	# Test wall-clock time for single runs with animation
	# sim.make(clean=True, animation=True); t_animation = wall_clock_test(n,m)

	# Test wall-clock time for single runs without animation
	sim.make(clean=True); t = wall_clock_test(n,m)

	# Test wall-clock time for batch runs, without animation
	sim.make(clean=True); t_batch = wall_clock_batch_test(n,m,batch)

	# Save
	if filename is not None: np.savez(filename, t=t,t_batch=t_batch)
	print("Finished everything")

def load(filename):
	##### Plot results of time tests #####
	data = np.load(filename+".npz")
	t = data['t'].astype(float)
	# t_animation = data['t_animation'].astype(float)
	t_batch = data['t_batch'].astype(float)
	return t, t_batch

def plot(filename,figurename=None):
	t, t_batch = load(filename)
	import matplotlib.pyplot as plt
	plt.plot(range(1,t.shape[0]+1),t.mean(axis=1),label="t")
	# plt.plot(range(1,t_animation.shape[0]+1),t_animation.mean(axis=1),label="t_animation")
	plt.plot(range(1,t_batch.shape[0]+1),t_batch.mean(axis=1)/5,label="t_batch")
	plt.xlabel("Number of robots")
	plt.ylabel("Wall-clock time")
	plt.legend()
	plt.savefig("time_plot.eps") if figurename is not None else plt.show()

def main():
	##### Primary input parameters -- Test parameters #####
	n = 50 # Max number of robots
	m = 5 # Re-runs to average out
	batch = 5 # Number of batch runs
	rtfactor = 300
	tl = 200
	filename = ("time_n%i_m%i_b%i_rt%i_tl%i" % (n,m,batch,rtfactor,tl))
	print("Writing to " + filename)
	##### Secondary input parameters -- Swarmulator configuration #####
	sim.runtime_setting("simulation_updatefreq", str("20"))
	sim.runtime_setting("environment", "square")
	sim.runtime_setting("animation_updatefreq", str("25"))
	sim.runtime_setting("simulation_realtimefactor", str(rtfactor))
	sim.runtime_setting("time_limit", str(tl))

	# run(n,m,batch,filename)
	plot(filename)

if __name__ == "__main__":
    main()
