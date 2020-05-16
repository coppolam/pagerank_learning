import numpy as np
from simulator import swarmulator
import time
from tqdm import tqdm
import matplotlib
matplotlib.rc('text', usetex=True)
import matplotlib.pyplot as plt
sim = swarmulator.swarmulator(verbose=True)
	
def wall_clock_test(n,m):
	t = np.zeros([n,m])
	for i in range(3,n+1): # Itarate over number of robots
		for j in range(0,m): # Do multiple runs to average out
			tic = time.time()    
			f = sim.run(i) # Launch simulation run
			toc = time.time()
			t[i-1,j] = toc-tic
			print("New result %i/%i, %i/%i: %.2f" % (i,n,j+1,m,t[i-1,j]))
	return t

def wall_clock_batch_test(n,m,batch):
	t = np.zeros([n,m])
	for i in tqdm(range(3,n+1)): # Itarate over number of robots
		for j in range(0,m): # Do multiple runs to average out
			tic = time.time()
			f = sim.batch_run(i,batch) # Launch batch
			toc = time.time()
			t[i-1,j] = (toc-tic)
	return t

def run(n,m,batch,filename=None):
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
	plt.clf()
	t, t_batch = load(filename)
	tmean = t.mean(axis=1)
	batchsize = 5
	tmean_batch = t_batch.mean(axis=1)/batchsize
	_ = plt.plot(range(1,t.shape[0]+1),tmean,color='blue',label="Single")
	_ = plt.plot(range(1,t_batch.shape[0]+1),tmean_batch,color='orange',label="Batch (5)")
	_ = plt.fill_between(range(1, len(tmean)+1),
			t.min(axis=1),
			t.max(axis=1),
			color='blue', alpha=0.2)
	_ = plt.fill_between(range(1, len(tmean)+1),
			t_batch.min(axis=1)/batchsize,
			t_batch.max(axis=1)/batchsize,
			color='orange', alpha=0.2)
	_ = plt.xlabel("Number of robots")
	_ = plt.ylabel("Wall-clock time [s]")
	_ = plt.legend(loc="upper left")
	_ = plt.savefig(figurename) if figurename is not None else plt.show()

def plot_rtfactor(filename,tl,figurename=None):
	plt.clf()
	t, t_batch = load(filename)
	tmean = t.mean(axis=1)/tl
	batchsize = 5
	tmean_batch = (t_batch.mean(axis=1)/batchsize)/tl
	f = plt.plot(range(1,t.shape[0]+1),1/tmean,color='blue',label="Single")
	f = plt.plot(range(1,t_batch.shape[0]+1),1/tmean_batch,color='orange',label="Batch (5)")
	# f = plt.fill_between(range(1, len(tmean)+1),
	# 		t.min(axis=1)/tl,
	# 		t.max(axis=1)/tl,
	# 		color='blue', alpha=0.2)
	# f = plt.fill_between(range(1, len(tmean)+1),
	# 		(t_batch.min(axis=1)/batchsize)/tl,
	# 		(t_batch.max(axis=1)/batchsize)/tl,
	# 		color='orange', alpha=0.2)
	f = plt.xlabel("Number of robots")
	f = plt.ylabel("Wall-clock time [s]")
	f = plt.legend(loc="upper left")
	f = plt.savefig(figurename) if figurename is not None else plt.show()

def main():
	##### Primary input parameters -- Test parameters #####
	n = 50 # Max number of robots
	m = 5 # Re-runs to average out
	batch = 5 # Number of batch runs
	rtfactor = 0
	tl = 200
	folder = "data/time/"
	filename = (folder+"time_n%i_m%i_b%i_rt%i_tl%i" % (n,m,batch,rtfactor,tl))
	print("Writing to " + filename)
	##### Secondary input parameters -- Swarmulator configuration #####
	sim.runtime_setting("simulation_updatefreq", str("20"))
	sim.runtime_setting("environment", "square")
	sim.runtime_setting("animation_updatefreq", str("25"))
	sim.runtime_setting("simulation_realtimefactor", str(rtfactor))
	sim.runtime_setting("time_limit", str(tl))

	# run(n,m,batch,filename)
	plot(filename,folder+"runtime_comparison.pdf")
	plot_rtfactor(filename,tl)

if __name__ == "__main__":
    main()
