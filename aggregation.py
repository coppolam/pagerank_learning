import datetime, subprocess, sys, random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import graphless_optimization as opt # Own package
from simulator import swarmulator # Own package
from tools import fileHandler as fh # Own package

class aggregation:
	def __init__(self, folder="../swarmulator"):
		'''Load simulator'''
		self.folder = folder
		self.data_folder = folder + "/logs/"
		self.run_id = str(random.randrange(100000))
		self.save_id = "data/" + self.run_id
		self.sim = swarmulator.swarmulator(folder) # Initialize sim
		
	def make(self, controller="controller_aggregation", agent="particle", clean=True, animation=False, logger=True, verbose=True):
		''' Build simulator'''
		self.sim.make(controller=controller, agent=agent, clean=clean, animation=animation, logger=logger, verbose=verbose) # Build (if already built, you can skip this)
		
	def run(self, policy="", logger_updatefreq=2, robots=30, time_limit=10000, realtimefactor=50, environment="square", run_id=None):
		''' Run simulator with specified settings '''
		subprocess.call("cd " + self.data_folder + " && rm *.csv", shell=True)
		self.sim.runtime_setting("time_limit", str(time_limit))
		self.sim.runtime_setting("simulation_realtimefactor", str(realtimefactor))
		self.sim.runtime_setting("logger_updatefreq", str(logger_updatefreq))
		self.sim.runtime_setting("environment", environment)
		self.sim.runtime_setting("policy", policy) # Use random policy
		self.robots = robots
		self.run_id = str(run_id) if run_id is not None else str(random.randrange(100000))
		print("Runtime ID: " + self.run_id)
		self.save_id = "data/" + self.run_id
		self.sim.run(robots,run_id=self.run_id) # Run it, and receive the fitness

	def save_learning_data(self,filename_ext=None):
		self.H = fh.read_matrix(self.data_folder,"H_"+self.sim.run_id)
		self.A = fh.read_matrix(self.data_folder,"A_"+self.sim.run_id)
		self.E = fh.read_matrix(self.data_folder,"E_"+self.sim.run_id)
		self.des = fh.read_matrix(self.data_folder,"des_"+self.sim.run_id)
		self.log = self.sim.load(id=self.sim.run_id)
		np.savez(self.save_id+"_learning_data"+filename_ext, des=self.des, H=self.H, A=self.A, E=self.E, log=self.log)
		print("Saved")

	def load(self,file=None):
		# if cmd is not None:
		# 	if np.size(cmd) == 1:
		# 		file = fh.get_latest_file('data/*_learning_data.npz')
		# 	else:
		# 		file = cmd[1]
		# 	print("Loading " + file[5:file.find('_learning_data')] + "_learning_data.npz")
		data = np.load(file)
		self.H = data['H'].astype(float)
		self.E = data['E'].astype(float)
		self.A = data['A'].astype(float)
		self.des = data['des'].astype(float)
		self.save_id = file[0:file.find('_learning_data')]
		self.log = data['log'].astype(float) #self.sim.load(file[5:file.find('_learning_data')])
		print("Loaded %s" %file)

	def optimize(self):
		p0 = np.ones([self.A.shape[1],int(self.A.max())]) / self.A.shape[1]

		temp = self.H + self.E
		empty_cols = np.where(~temp.any(axis=0))[0]
		empty_rows = np.where(~temp.any(axis=1))[0]
		empty_states = np.intersect1d(empty_cols,empty_rows,assume_unique=True)
		self.des = np.zeros([1,16])[0]
		self.des[15] = 1
		print(self.des)
		self.result, self.policy, self.empty_states = opt.main(p0, self.des, self.H, self.A, self.E)
		print("Unknown states:" + str(self.empty_states))
		print('{:=^40}'.format(' Optimization '))
		print("Final fitness: " + str(self.result.fun))
		print("[ policy ]")
		np.set_printoptions(threshold=sys.maxsize)
		print(self.policy)

	def save_optimized(self):
		np.savez(self.save_id+"_optimization", result=self.result, policy=self.policy, empty_states=self.empty_states)

	def disp(self):
		print(self.H)
		print(self.E)
		print(self.A)
		print("States desireability: ", str(self.des))
		
	def benchmark(self, controller=None, agent=None, robots=30, time_limit=1000, realtimefactor=50, environment="square",runs=100,policy=None):
		self.sim.make(controller=controller,agent=agent,clean=True, animation=True, logger=False, verbose=True)
		self.sim.runtime_setting("time_limit", str(time_limit))
		self.sim.runtime_setting("simulation_realtimefactor", str(realtimefactor))
		self.sim.runtime_setting("environment", environment)
		# TODO: Change to batch run
		# Benchmark
		f_0 = []
		if policy is None: self.sim.runtime_setting("policy", "") # Use random policy
		else: self.sim.runtime_setting("policy",policy)
		for i in range(0,runs):
			print('{:=^40}'.format(' Simulator run '))
			print("Run " + str(i) + "/" + str(runs))
			f_0 = np.append(f_0,self.sim.run(robots))

		# Optimize
		f_n = []
		policy_file = self.sim.path + "/conf/state_action_matrices/aggregation_policy_evolved.txt"
		fh.save_to_txt(self.policy, policy_file)
		self.sim.runtime_setting("policy", policy_file) # Use random policy
		for i in range(0,runs):
			print('{:=^40}'.format(' Simulator run '))
			print("Run " + str(i) + "/"  +str(runs))
			f_n = np.append(f_n,self.sim.run(robots))
		np.savez(self.save_id+"_validation", f_n=f_n)

	def histplots(self, filename=None):
		# Load
		file = filename if filename is not None else self.save_id
		data_validation = np.load(file + "_validation.npz")
		fitness_0 = data_validation['f_0'].astype(float)
		fitness_n = data_validation['f_n'].astype(float)

		# Plot
		matplotlib.rcParams['text.usetex'] = False # True for latex style
		alpha = 0.5;
		plt.hist(fitness_0, alpha=alpha, density=True, label='$\pi_0$')
		plt.hist(fitness_n, alpha=alpha, density=True, label='$\pi_n$')
		plt.xlabel("Fitness")
		plt.ylabel("Instances")
		plt.legend(loc='upper right')
		plt.savefig(self.save_id+"_histplot.png")

		
	## Re-evaluating
	def reevaluate(self,*args):
		'''Re-evaluate the fitnesses based on new fitness functions'''
		id_column = 1
		robots = int(self.log[:,id_column].max())
		time_column = 0
		t = np.unique(self.log[:,0])
		f_official = np.zeros(t.shape)
		fitness = np.zeros([t.size,len(args)])
		arguments = locals()
		print("Re-evaluating")
		a = 0
		states = np.zeros([t.size,robots])
		des = np.zeros([t.size,self.H.shape[0]])
		for step in t:
			d = self.log[np.where(self.log[:,time_column] == step)]
			fref = 0
			for i in args:
				fitness[a,fref] = i(d)
				fref += 1
			f_official[a] = d[:,5].astype(float).mean()
			states[a] = d[0:robots,4].astype(int)
			for r in np.arange(0,np.max(states[a])+1).astype(int):
				if r < self.H.shape[0]: # Guard for max state in case inconsistent with Swarmulator
					des[a,r] = np.count_nonzero(states[a] == r)
			a += 1
		print("Re-evaluation done")
		return t, fitness, des

	## Fitnesses
	def plot_fitness(self,t,fitness):
		for a in range(fitness.shape[1]):
			plt.plot(t,fitness[:,a]/np.mean(fitness[:,a]))
		plt.ylabel("Fitness")
		plt.xlabel("Time [s]")
		plt.show()

	## Correlation
	def plot_correlation(self,fitness):
		for a in range(1,fitness.shape[1]):
			plt.plot(fitness[:,0],fitness[:,a],'*')
			c = np.corrcoef(fitness[:,0],fitness[:,a])[0,1]
			print("Cov 0:", str(a), " = ", str(c))
		plt.ylabel("Local Fitness")
		plt.xlabel("Global Fitness")
		plt.show()