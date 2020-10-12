import datetime, time, subprocess, sys, random, glob, os
import numpy as np
from tqdm import tqdm

from . import pagerank_evolve as opt
from . import desired_states_extractor
from tools import swarmulator
from tools import fileHandler as fh
from tools import matrixOperations as matop

class simulator:
	''' 
	Higher level API to interact with simulator
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
		
	def make(self, controller, agent, 
				clean=True, animation=False, logger=True, verbose=True):
		''' Build simulator with the desired settings'''
		# Build (if already built, you can skip this)
		self.sim.make(controller=controller, 
						agent=agent, 
						clean=clean,
						animation=animation, 
						logger=logger, 
						verbose=verbose)

	def save_policy(self, policy, pr_actions=None, name="temp"):
		'''Save the policy in the correct format for use in Swarmulator'''

		# Resize policy to correct dimensions and normalize,
		# else assume it's already correct.
		if pr_actions is not None:
			policy = np.reshape(policy,(policy.size//pr_actions,pr_actions))
			# Normalize rows if needed
			if pr_actions > 1:
				policy = matop.normalize_rows(policy)

		# Save the policy so it can be used by the simulator
		policy_filename = "conf/policies/%s.txt"%name
		policy_file = self.sim.path + "/" + policy_filename
		
		# Write in the correct format for reading
		if policy.shape[1] == 1:
			fh.save_to_txt(policy.T, policy_file)
		else:
			fh.save_to_txt(policy, policy_file)

		# Return the filename
		return policy_filename

	def save_log(self,filename_ext=None):
		'''Save a logfile'''
		
		# Load the log from the txt file
		filename = self.logs_folder + "log_" + str(self.sim.run_id) + ".txt"
		self.log = self.sim.load(file=filename)
		
		# Set up filename in the folder
		save_filename = self.savefolder+filename_ext

		# Save
		np.savez(save_filename, log=self.log)

		# Confirmation
		print("Saved to %s"%save_filename)

	def save_learning_data(self,filename_ext=None):
		'''Save the models from a log file'''
		
		# Read the H matrix
		H = fh.read_matrix(self.logs_folder,"H_"+self.sim.run_id)

		# Read the A matrices
		A = []
		for i in range(len(glob.glob(
						self.logs_folder+"A_"+self.sim.run_id+"_*"))):
						A.append(fh.read_matrix(
						self.logs_folder,"A_"+self.sim.run_id+"_"+str(i)))
		
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

	def load(self,file,verbose=True):
		''' Load the model from a swarmulator log file '''

		# Load all data from the npz file
		data = np.load(file)

		# Load the A matrices
		self.A = data['A'].astype(float)

		# H matrix = sum of all A matrices
		self.H = np.sum(self.A, axis=0)

		# E matrix
		self.E = data['E'].astype(float)

		# Run log
		self.log = data['log'].astype(float)

		# Print to terminal
		if verbose:
			print("Loaded %s (from %s)" 
				% (file,
				datetime.datetime.fromtimestamp(os.path.getmtime(file))))

	def load_update(self,file,discount=1.0,verbose=False):
		''' 
		Load the model from a swarmulator log file 
		on top of the existing one
		'''

		# Load all data from the npz file
		data = np.load(file)
		
		# Load the A matrices (to temp list Am)
		Am = data['A'].astype(float)

		# Update A matrices
		for i in range(self.A.shape[0]):
			self.A[i] = discount*self.A[i] + Am[i]

		# Update H matrix
		self.H = np.sum(self.A, axis=0)

		# Update E matrix
		self.E = discount*self.E + data['E'].astype(float)

		# Print to terminal
		if verbose:
			print("Loaded %s (from %s)"
				% (file,
				datetime.datetime.fromtimestamp(os.path.getmtime(file))))

	def save_optimization_data(self,policy,des,filename_ext=None):
		# Load up sim variables locally
		A = self.A
		E = self.E
		H0 = np.sum(A, axis=0)
		with np.errstate(divide='ignore',invalid='ignore'):
			r = H0.sum(axis=1) / E.sum(axis=1)
			r = np.nan_to_num(r) # Remove NaN Just in case
		alpha = r / (1 + r)

		# Get optimized policy
		o = opt.pagerank_evolve(des,self.A,self.E)
		H1 = o.update_H(A, policy)
		del o

		# Save 
		save_filename = self.savefolder+filename_ext
		np.savez(save_filename, H0=H0, H1=H1, A=A, E=E, 
									policy=policy, alpha=alpha, des=des)

	def optimize(self, p0, iterations=0, model=None, settings=None, debug=True):
		'''Optimize the policy based on the desired states'''
		i = 0
		
		# Get the desired states using the trained feed-forward network
		dse = desired_states_extractor.desired_states_extractor()
		
		## Load a neural network model if needed
		## If a model is not specified, try to load it
		if model is not None:
			dse.network = model
		else:
			dse.load_model("data/%s/models.pkl"%\
						settings["controller"], modelnumber=499)			
		
		## Get desired states
		des = dse.get_des(dim=settings["pr_states"])

		# Initialize and run the optimizer
		o = opt.pagerank_evolve(des,self.A,self.E)
		policy = o.run(p0,generations=500,plot=debug)
		del o

		# Save
		self.save_optimization_data(policy,des,"optimization_%i"%i)

		return policy

	def disp(self):
		'''Display the model to the terminal'''
		
		# Print the H matrix
		print("H:\n",self.H)

		# Print the E matrix
		print("E:\n",self.E)
		
		# Print all matrices in the set A
		for i,a in enumerate(self.A):
			print("A%i:"%i)
			print(a)

	def run(self, pr_states=0, pr_actions=0, policy_filename="", fitness="", 
		logger_updatefreq=2, robots=30, time_limit=10000, 
		realtimefactor=300, environment="square20", run_id=None,**kwargs):
		''' Run simulator with specified settings '''
		
		# Clear out the folder
		subprocess.call("cd " + self.logs_folder + " && rm *.csv", shell=True)

		# Set all the relevant runtime settings
		self.sim.runtime_setting("time_limit", str(time_limit))
		self.sim.runtime_setting("simulation_realtimefactor",
												str(realtimefactor))
		self.sim.runtime_setting("logger_updatefreq", str(logger_updatefreq))
		self.sim.runtime_setting("environment", environment)
		self.sim.runtime_setting("policy", policy_filename)
		self.sim.runtime_setting("fitness", fitness)
		self.sim.runtime_setting("pr_states", str(pr_states))
		self.sim.runtime_setting("pr_actions", str(pr_actions))

		# Set up the number of robots
		self.robots = robots

		# Set up the runtime ID, if not already set
		self.run_id = str(run_id) \
			if run_id is not None \
			else str(random.randrange(100000))
		print("Runtime ID: %s" % self.run_id)

		# Set up the save_id
		self.save_id = self.savefolder + self.run_id

		# Run it
		f = self.sim.run(robots,run_id=self.run_id)

		# Return fitness
		return float(f)

	def benchmark(self, policy, controller, agent, fitness, robots=30, 
		time_limit=1000, realtimefactor=300, environment="square20", 
		runs=100, make=True, pr_states=0, pr_actions=0,**kwargs):
		'''Perform many runs of the simulator to benchmark the behavior'''
		
		# Save policy file to test
		policy_file = self.save_policy(policy)

		# Build with correct settings
		if make == True:
			self.sim.make(controller=controller, agent=agent,
				clean=True, animation=False, logger=True, verbose=False)

		# Set the runtime settings
		self.sim.runtime_setting("time_limit", str(time_limit))
		self.sim.runtime_setting("simulation_realtimefactor", 
										str(realtimefactor))
		self.sim.runtime_setting("environment", environment)
		self.sim.runtime_setting("policy", policy_file)
		self.sim.runtime_setting("fitness", fitness)
		self.sim.runtime_setting("pr_states", str(pr_states))
		self.sim.runtime_setting("pr_actions", str(pr_actions))

		# Run one by one, comment out and uncomment above for batches
		f = []
		for i in range(0,runs):
			# Run it, and receive the fitness
			f = np.append(f, self.sim.run(robots,run_id=self.run_id))

			# Print the fitness list, just to show progress
			print(f)

			# Save the log
			self.save_log(filename_ext="sample_log_%i"%i)

		return f

	def observe(self, controller, agent, policy, fitness, robots=30, 
		time_limit=0, realtimefactor=300, environment="square20",**kwargs):
		'''
		Launch a single run of the simulator with 
		animation to observe what happens
		'''

		# Save policy file to test
		policy_file = self.save_policy(policy)

		# Build with correct settings (Animation=ON!)
		self.sim.make(controller, agent,
			clean=True, animation=True, logger=False, verbose=False)
		
		# Runtime settings
		self.sim.runtime_setting("time_limit", str(time_limit))
		self.sim.runtime_setting("simulation_realtimefactor", 
										str(realtimefactor))
		self.sim.runtime_setting("environment", environment)
		self.sim.runtime_setting("policy", policy_file)
		self.sim.runtime_setting("fitness", fitness)
		self.sim.runtime_setting("pr_states", str(0)) # 0 = No PR estimator
		self.sim.runtime_setting("pr_actions", str(0)) # 0 = No PR estimator

		# Run
		self.sim.run(robots)
		return True

	def extract(self):
		''' 
		Extract data from the log file that has already been 
		loaded using the load method
		'''
		
		# The 0th column is the time column, the 1st column is the IDs
		time_column = 0
		id_column = 1
		
		# Get the time vector
		t = np.unique(self.log[:,time_column])

		# Get the number of robots
		robots = int(self.log[:,id_column].max())

		# Set up empty arrays where we will store the fitness and the states
		fitness = np.zeros(t.shape)
		states = np.zeros([t.size,robots])
		states_count = np.zeros([t.size,self.H.shape[0]])
		
		# Extract what is relevant from each log 
		for a, step in enumerate(t):
			d = self.log[np.where(self.log[:,time_column] == step)]
			fitness[a] = d[:,5].astype(float).mean()
			states[a] = d[0:robots,4].astype(int)

			for r in np.arange(0,np.max(states[a])+1).astype(int):
				# Guard for max state in case inconsistent with Swarmulator
				if r < self.H.shape[0]:
					states_count[a,r] = np.count_nonzero(states[a] == r)
		
		# Return it
		return t, states_count, fitness