import datetime, time, subprocess, sys, random, glob, os
import numpy as np
from tqdm import tqdm

from simulators import swarmulator
from . import pagerank_evolve as opt
from tools import fileHandler as fh
from tools import matrixOperations as matop

class simulator:
	''' Higher level API to interact with simulator
		Set up to operate with Swarmulator for this particular project
	'''
	def __init__(self, folder="../swarmulator", savefolder="data/"):
		'''Load simulator'''

		# Simulator folder
		self.folder = folder

		# Load logs folder in simulator
		self.logs_folder = folder + "/logs/"

		# Set up a runtime ID for a simulation
		self.run_id = str(random.randrange(100000))

		# Set up a folder which will be used to save the outputs
		# and set up the directory for the folder if it doesn't exist
		self.savefolder = savefolder
		directory = os.path.dirname(self.savefolder)
		if not os.path.exists(directory):
			os.makedirs(directory)

		self.save_id = self.savefolder + self.run_id
		
		# Initialize simulator
		self.sim = swarmulator.swarmulator(folder,verbose=False)
		
	def make(self, controller, agent, clean=True, animation=False, logger=True, verbose=True):
		''' Build simulator with the desired settings'''
		# Build (if already built, you can skip this)
		self.sim.make(controller=controller, 
						agent=agent, 
						clean=clean,
						animation=animation, 
						logger=logger, 
						verbose=verbose)
		
	def run(self, pr_states=0, pr_actions=0, policy="", fitness="", logger_updatefreq=2, robots=30, time_limit=10000, realtimefactor=300, environment="square20", run_id=None):
		''' Run simulator with specified settings '''
		
		# Clear out the folder
		subprocess.call("cd " + self.logs_folder + " && rm *.csv", shell=True)

		# Set all the relevant runtime settings
		self.sim.runtime_setting("time_limit", str(time_limit))
		self.sim.runtime_setting("simulation_realtimefactor", str(realtimefactor))
		self.sim.runtime_setting("logger_updatefreq", str(logger_updatefreq))
		self.sim.runtime_setting("environment", environment)
		self.sim.runtime_setting("policy", policy)
		self.sim.runtime_setting("fitness", fitness)
		self.sim.runtime_setting("pr_states", str(pr_states))
		self.sim.runtime_setting("pr_actions", str(pr_actions))

		# Set up the number of robots
		self.robots = robots

		# Set up the runtime ID, if not already set
		self.run_id = str(run_id) if run_id is not None else str(random.randrange(100000))
		print("Runtime ID: %s"%self.run_id)

		# Set up the save_id
		self.save_id = self.savefolder + self.run_id

		# Run it
		self.sim.run(robots,run_id=self.run_id)

	def save_learning_data(self,filename_ext=None):
		'''Save the models from a log file'''
		# Read the H matrix
		H = fh.read_matrix(self.logs_folder,"H_"+self.sim.run_id)

		# Read the A matrices
		A = []
		for i in range(len(glob.glob(self.logs_folder+"A_"+self.sim.run_id+"_*"))):
			A.append(fh.read_matrix(self.logs_folder,"A_"+self.sim.run_id+"_"+str(i)))

		# Read the E matrix
		E = fh.read_matrix(self.logs_folder,"E_"+self.sim.run_id)

		# Load the logfile
		log = self.sim.load(file=self.logs_folder+"log_"+self.sim.run_id+".txt")

		# Set up name
		save_filename = self.savefolder+filename_ext

		# Save the data
		np.savez(save_filename, H=H, A=A, E=E, log=log)
		
		# Confirmation
		print("Saved to %s"%save_filename)
		
		return save_filename

	def save_log(self,filename_ext=None):
		'''Save a logfile'''
		# Load the log from the txt file
		self.log = self.sim.load(file=self.logs_folder+"log_"+str(self.sim.run_id)+".txt")
		
		# Set up filename in the folder
		save_filename = self.savefolder+filename_ext

		# Save
		np.savez(save_filename, log=self.log)

		# Confirmation
		print("Saved to %s"%save_filename)

	def load(self,file,verbose=True):
		'''Load the model from a swarmulator log file'''
		# Load all data from the npz file
		data = np.load(file)

		# 
		self.A = data['A'].astype(float)
		self.H = np.sum(self.A, axis=0)
		self.E = data['E'].astype(float)
		self.log = data['log'].astype(float)
		if verbose: print("Loaded %s (from %s)" % (file,datetime.datetime.fromtimestamp(os.path.getmtime(file))))

	def load_update(self,file,discount=1.0,verbose=False):
		'''Load the model from a swarmulator log file on top of the existing one'''
		data = np.load(file)
		Am = data['A'].astype(float)
		for i in range(self.A.shape[0]): self.A[i] = discount*self.A[i] + Am[i]
		self.E = discount*self.E + data['E'].astype(float)
		self.H = np.sum(self.A, axis=0)
		if verbose: print("Loaded %s (from %s)" %(file,datetime.datetime.fromtimestamp(os.path.getmtime(file))))

	def optimize(self, p0, des, debug=True):
		'''Optimize the policy based on the desired states'''
		
		# Initialize and run optimizer
		o = opt.pagerank_evolve(des,self.A,self.E)
		policy = o.run(p0)

		# For analysis/debug purposes, show states that have not been visited
		if debug is True:
			print(policy)
			temp = self.H + self.E
			empty_cols = np.where(~temp.any(axis=0))[0]
			empty_rows = np.where(~temp.any(axis=1))[0]
			empty_states = np.intersect1d(empty_cols,empty_rows,assume_unique=True)
			print("\nUnknown states: \t" + str(empty_states))
			np.set_printoptions(threshold=sys.maxsize)
	
		return policy

	def disp(self):
		'''Display the model to the terminal'''
		
		# Print the H matrix
		print("H:\n",self.H)

		# Print the E matrix
		print("E:\n",self.E)
		
		# Print all matrices in the set A
		for i,a in enumerate(self.A):
			print("A%i:"%i); print(a);

	def benchmark(self, controller, agent, policy, fitness, robots=30, 
		time_limit=1000, realtimefactor=300, environment="square20", runs=100, make=True, pr_states=0, pr_actions=0):
		'''Perform many runs of the simulator to benchmark the behavior'''
		
		# Save policy file to test
		policy_file = self.sim.path + "/conf/policies/aggregation_policy_benchmark.txt"
		if policy.shape[1] == 1: fh.save_to_txt(policy.T, policy_file) # Number of columns = 1
		else: fh.save_to_txt(policy, policy_file)

		# Build with correct settings
		if make == True:
			self.sim.make(controller=controller, agent=agent, clean=True, animation=False, logger=True, verbose=False)
		self.sim.runtime_setting("time_limit", str(time_limit))
		self.sim.runtime_setting("simulation_realtimefactor", str(realtimefactor))
		self.sim.runtime_setting("environment", environment)
		self.sim.runtime_setting("policy", policy_file)
		self.sim.runtime_setting("fitness", fitness)
		self.sim.runtime_setting("pr_states", str(pr_states)) # Don't run pr estimator
		self.sim.runtime_setting("pr_actions", str(pr_actions)) # Don't run pr estimator

		f = []
		# for i in tqdm(range(0,round(runs/5))): # Run in batches for speed, uncomment this if you want it
		# 	f = np.append(f,self.sim.batch_run(robots,5))
		# 	print(f)
		for i in range(0,runs): # Run one by one, comment out and uncomment above for batches
			f = np.append(f, self.sim.run(robots,run_id=self.run_id)) # Run it, and receive the fitness
			print(f)
			self.save_log(filename_ext="sample_log_%i"%i)

		return f

	def observe(self, controller, agent, policy, fitness, robots=30, 
		time_limit=0, realtimefactor=300, environment="square"):
		'''Launch a single run of the simulator with animation to observe what happens'''

		# Save policy file to test
		policy_file = self.sim.path + "/conf/policies/aggregation_policy_benchmark.txt"
		if policy.shape[1] == 1: fh.save_to_txt(policy.T, policy_file) # Number of columns = 1
		else: fh.save_to_txt(policy, policy_file)

		# Build with correct settings
		self.sim.make(controller,agent,clean=True, animation=True, logger=True, verbose=True)
		self.sim.runtime_setting("time_limit", str(time_limit))
		self.sim.runtime_setting("simulation_realtimefactor", str(realtimefactor))
		self.sim.runtime_setting("environment", environment)
		self.sim.runtime_setting("policy", policy_file) # Use random policy
		self.sim.runtime_setting("fitness", fitness)
		self.sim.runtime_setting("pr_states", str(0)) # Don't run pr estimator
		self.sim.runtime_setting("pr_actions", str(0)) # Don't run pr estimator

		# Run
		self.sim.run(robots)
		return True
	
	def extract(self):
		''' Extract data from the log file that has already been loaded using the load method'''
		time_column = 0
		id_column = 1
		t = np.unique(self.log[:,time_column])
		robots = int(self.log[:,id_column].max())
		fitness = np.zeros(t.shape)
		states = np.zeros([t.size,robots])
		states_count = np.zeros([t.size,self.H.shape[0]])
		for a, step in enumerate(t): # Extract what is relevant from each log 
			d = self.log[np.where(self.log[:,time_column] == step)]
			fitness[a] = d[:,5].astype(float).mean()
			states[a] = d[0:robots,4].astype(int)
			for r in np.arange(0,np.max(states[a])+1).astype(int):
				if r < self.H.shape[0]: # Guard for max state in case inconsistent with Swarmulator
					states_count[a,r] = np.count_nonzero(states[a] == r)
		return t, states_count, fitness
