import datetime, subprocess, sys, random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import graphless_optimization as opt # Own package
from simulator import swarmulator # Own package
from tools import fileHandler as fh # Own package
from tqdm import tqdm
from tools import matrixOperations as matop

class aggregation:
	def __init__(self, folder="../swarmulator"):
		'''Load simulator'''
		self.folder = folder
		self.data_folder = folder + "/logs/"
		self.run_id = str(random.randrange(100000))
		self.save_id = "data/" + self.run_id
		self.sim = swarmulator.swarmulator(folder,verbose=False) # Initialize sim
		
	def make(self, controller, agent, clean=True, animation=False, logger=True, verbose=True):
		''' Build simulator'''
		self.sim.make(controller=controller, agent=agent, clean=clean, animation=animation, logger=logger, verbose=verbose) # Build (if already built, you can skip this)
		
	def run(self, policy="", logger_updatefreq=2, robots=30, time_limit=10000, realtimefactor=300, environment="square", run_id=None):
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
		self.A = []
		for i in range(0,self.H.shape[0]):
			self.A.append(fh.read_matrix(self.data_folder,"A_"+self.sim.run_id+"_"+str(i)))
		self.E = fh.read_matrix(self.data_folder,"E_"+self.sim.run_id)
		self.des = fh.read_matrix(self.data_folder,"des_"+self.sim.run_id)
		self.log = self.sim.load(file=self.data_folder+"log_"+self.sim.run_id+".txt") # Latest
		np.savez(self.save_id+"_learning_data"+filename_ext, des=self.des, H=self.H, A=self.A, E=self.E, log=self.log)
		print("Saved")

	def load(self,file=None):
		data = np.load(file)
		self.H = data['H'].astype(float)
		self.E = data['E'].astype(float)
		self.A = data['A'].astype(float)
		self.des = data['des'].astype(float)
		self.save_id = file[0:file.find('_learning_data')]
		self.log = data['log'].astype(float) #self.sim.load(file[5:file.find('_learning_data')])
		print("Loaded %s" %file)

	def optimize(self,des):
		p0 = np.ones([self.A.shape[1],int(self.A.max())]) / self.A.shape[1]

		temp = self.H + self.E
		empty_cols = np.where(~temp.any(axis=0))[0]
		empty_rows = np.where(~temp.any(axis=1))[0]
		empty_states = np.intersect1d(empty_cols,empty_rows,assume_unique=True)
		self.result, policy, self.empty_states = opt.main(p0, des, self.H, self.A, self.E)
		print("Unknown states:" + str(self.empty_states))
		print('{:=^40}'.format(' Optimization '))
		print("Final fitness: " + str(self.result.fun))
		print("[ policy ]")
		print(policy)
		np.set_printoptions(threshold=sys.maxsize)
		return policy

	def save_optimized(self):
		np.savez(self.save_id+"_optimization", result=self.result, policy=self.policy, empty_states=self.empty_states)

	def disp(self):
		print(self.H)
		print(self.E)
		print(self.A)
		print("States desireability: ", str(self.des))
		
	def benchmark(self, policy, controller, agent, robots=30, time_limit=1000, realtimefactor=0, environment="square",runs=100):
		self.sim.make(controller=controller,agent=agent,clean=True, animation=False, logger=False, verbose=False)
		self.sim.runtime_setting("time_limit", str(time_limit))
		self.sim.runtime_setting("simulation_realtimefactor", str(realtimefactor))
		self.sim.runtime_setting("environment", environment)
		
		# Optimize
		policy_file = self.sim.path + "/conf/state_action_matrices/aggregation_policy_benchmark.txt"
		fh.save_to_txt(policy, policy_file)
		self.sim.runtime_setting("policy", policy_file) # Use random policy
		f = []
		for i in tqdm(range(0,round(runs/5))):
			f = np.append(f,self.sim.batch_run(robots,5))
		return f

	def observe(self, policy, clean=True, controller=None, agent=None, robots=30, time_limit=0, realtimefactor=50, environment="square",runs=100):
		self.sim.make(controller=controller,agent=agent,clean=clean, animation=True, logger=True, verbose=False)
		self.sim.runtime_setting("time_limit", str(time_limit))
		self.sim.runtime_setting("simulation_realtimefactor", str(realtimefactor))
		self.sim.runtime_setting("environment", environment)
		
		# Optimize
		policy_file = self.sim.path + "/conf/state_action_matrices/policy_observe.txt"
		fh.save_to_txt(policy, policy_file)
		self.sim.runtime_setting("policy", policy_file) # Use random policy
		f = []
		self.sim.run(robots)
		log = self.sim.load(file=self.data_folder+"log_"+str(self.sim.run_id)+".txt") # Latest
		return log

	def histplots(self, filename=None):
		# Load
		file = filename if filename is not None else self.save_id
		data_validation = np.load(file + "_validation.npz")
		fitness_0 = data_validation['f_0'].astype(float)
		fitness_n = data_validation['f_n'].astype(float)

		# Plot
		matplotlib.rcParams['text.usetex'] = False # True for latex style
		alpha = 0.5;
		plt.hist(fitness_0, alpha=alpha, label='$\pi_0$')
		plt.hist(fitness_n, alpha=alpha, label='$\pi_n$')
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
		for step in tqdm(t):
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
		return t, fitness, des, f_official

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