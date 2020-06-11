import datetime, time, subprocess, sys, random, glob, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from classes import pagerank_optimization as opt
from simulators import swarmulator
from tools import fileHandler as fh
from tools import matrixOperations as matop

class simulator:
	def __init__(self, folder="../swarmulator", savefolder="data/"):
		'''Load simulator'''
		self.folder = folder
		self.logs_folder = folder + "/logs/"
		self.run_id = str(random.randrange(100000))
		self.savefolder = savefolder
		directory = os.path.dirname(self.savefolder)
		if not os.path.exists(directory): os.makedirs(directory)
		self.save_id = self.savefolder + self.run_id
		self.sim = swarmulator.swarmulator(folder,verbose=False) # Initialize sim
		
	def make(self, controller, agent, clean=True, animation=False, logger=True, verbose=True):
		''' Build simulator'''
		self.sim.make(controller=controller, agent=agent, clean=clean, animation=animation, logger=logger, verbose=verbose) # Build (if already built, you can skip this)
		
	def run(self, pr_states=0, pr_actions=0, policy="", fitness="", logger_updatefreq=2, robots=30, time_limit=10000, realtimefactor=300, environment="square", run_id=None):
		''' Run simulator with specified settings '''
		subprocess.call("cd " + self.logs_folder + " && rm *.csv", shell=True)
		self.sim.runtime_setting("time_limit", str(time_limit))
		self.sim.runtime_setting("simulation_realtimefactor", str(realtimefactor))
		self.sim.runtime_setting("logger_updatefreq", str(logger_updatefreq))
		self.sim.runtime_setting("environment", environment)
		self.sim.runtime_setting("policy", policy) # Use random policy
		self.sim.runtime_setting("fitness", fitness)
		self.sim.runtime_setting("pr_states", str(pr_states))
		self.sim.runtime_setting("pr_actions", str(pr_actions))
		self.robots = robots
		self.run_id = str(run_id) if run_id is not None else str(random.randrange(100000))
		print("Runtime ID: %s"%self.run_id)
		self.save_id = self.savefolder + self.run_id
		self.sim.run(robots,run_id=self.run_id) # Run it, and receive the fitness

	def save_learning_data(self,filename_ext=None):
		self.H = fh.read_matrix(self.logs_folder,"H_"+self.sim.run_id)
		self.A = []
		for i in range(len(glob.glob(self.logs_folder+"A_"+self.sim.run_id+"_*"))):
			self.A.append(fh.read_matrix(self.logs_folder,"A_"+self.sim.run_id+"_"+str(i)))
		self.E = fh.read_matrix(self.logs_folder,"E_"+self.sim.run_id)
		self.log = self.sim.load(file=self.logs_folder+"log_"+self.sim.run_id+".txt")
		save_filename = self.savefolder+filename_ext
		np.savez(save_filename, H=self.H, A=self.A, E=self.E, log=self.log)
		print("Saved to %s"%save_filename)

	def load(self,file):
		data = np.load(file)
		self.H = data['H'].astype(float)
		self.E = data['E'].astype(float)
		self.A = data['A'].astype(float)
		self.log = data['log'].astype(float)
		print("Loaded %s (from %s)" %(file,datetime.datetime.fromtimestamp(os.path.getmtime(file))))

	def optimize(self, p0, des):
		temp = self.H + self.E
		empty_cols = np.where(~temp.any(axis=0))[0]
		empty_rows = np.where(~temp.any(axis=1))[0]
		empty_states = np.intersect1d(empty_cols,empty_rows,assume_unique=True)
		self.result, policy, self.empty_states = opt.main(p0, des, self.H, self.A, self.E)
		print("\nUnknown states: \t" + str(self.empty_states))
		print("Final fitness: \t" + str(self.result.fun))
		print("Policy: \n" + str(policy))
		np.set_printoptions(threshold=sys.maxsize)
		return policy

	def disp(self):
		print("H:"); print(self.H)
		print("E:"); print(self.E)
		for i,a in enumerate(self.A): print("A%i:"%i); print(a);

	def benchmark(self, controller, agent, policy, fitness, robots=30, 
		time_limit=1000, realtimefactor=300, environment="square",runs=100):
		'''Perform many runs of the simulator to observe what happens'''
		
		#### Save policy file to test ####
		policy_file = self.sim.path + "/conf/state_action_matrices/aggregation_policy_benchmark.txt"
		if policy.shape[1] == 1: fh.save_to_txt(policy.T, policy_file) # Number of columns = 1
		else: fh.save_to_txt(policy, policy_file)

		#### Build with correct settings ####
		self.sim.make(controller=controller, agent=agent, clean=True, animation=False, logger=False, verbose=False)
		self.sim.runtime_setting("time_limit", str(time_limit))
		self.sim.runtime_setting("simulation_realtimefactor", str(realtimefactor))
		self.sim.runtime_setting("environment", environment)
		self.sim.runtime_setting("policy", policy_file)
		self.sim.runtime_setting("fitness", fitness)
		self.sim.runtime_setting("pr_states", str(0)) # Don't run pr estimator
		self.sim.runtime_setting("pr_actions", str(0)) # Don't run pr estimator

		#### Run (in batches for speed) ####
		f = []
		for i in tqdm(range(0,round(runs/5))):
			f = np.append(f,self.sim.batch_run(robots,5))
			print(f)
		return f

	def observe(self, controller, agent, policy, clean=True, robots=30, 
		time_limit=0, realtimefactor=300, environment="square"):
		'''Launch a single run of the simulator with animation to observe what happens'''

		#### Save policy file to test ####
		policy_file = self.sim.path + "/conf/state_action_matrices/aggregation_policy_benchmark.txt"
		if policy.shape[1] == 1: fh.save_to_txt(policy.T, policy_file) # Number of columns = 1
		else: fh.save_to_txt(policy, policy_file)

		#### Build with correct settings ####
		self.sim.make(controller,agent,clean=True, animation=True, logger=False, verbose=True)
		self.sim.runtime_setting("time_limit", str(time_limit))
		self.sim.runtime_setting("simulation_realtimefactor", str(realtimefactor))
		self.sim.runtime_setting("environment", environment)
		self.sim.runtime_setting("policy", policy_file) # Use random policy
		self.sim.runtime_setting("pr_states", str(0)) # Don't run pr estimator
		self.sim.runtime_setting("pr_actions", str(0)) # Don't run pr estimator

		self.sim.run(robots)
		# log = self.sim.load(file=self.logs_folder+"log_"+str(self.sim.run_id)+".txt") # Latest
		# return log
	
	def extract(self):
		''' Extract data from the log file that has already been loaded using the load method'''

		time_column = 0
		id_column = 1
		t = np.unique(self.log[:,time_column])
		robots = int(self.log[:,id_column].max())
		fitness = np.zeros(t.shape)
		a = 0
		states = np.zeros([t.size,robots])
		states_count = np.zeros([t.size,self.H.shape[0]])
		for step in tqdm(t): # Extract what is relevant from each log 
			d = self.log[np.where(self.log[:,time_column] == step)]
			fitness[a] = d[:,5].astype(float).mean()
			states[a] = d[0:robots,4].astype(int)
			for r in np.arange(0,np.max(states[a])+1).astype(int):
				if r < self.H.shape[0]: # Guard for max state in case inconsistent with Swarmulator
					states_count[a,r] = np.count_nonzero(states[a] == r)
			a += 1
		return t, states_count, fitness