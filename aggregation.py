import datetime, subprocess, sys
import numpy as np
from simulator import swarmulator # Own package
from tools import fileHandler as fh # Own package
import matplotlib
import matplotlib.pyplot as plt
import graphless_optimization as opt

class aggregation:
	def __init__(self, folder="../swarmulator"):
		self.folder = folder
		self.data_folder = folder + "/logs/"
		self.save_id = "data/" + str(datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S"))
		self.sim = swarmulator.swarmulator(folder) # Initialize sim
		
	def run(self,robots=30, time_limit=10000, realtimefactor=50, environment="square"):
		subprocess.call("cd " + self.data_folder + " && rm *.csv", shell=True)
		# self.sim.make(clean=True, animation=False, logger=True, verbose=True) # Build (if already built, you can skip this)
		self.sim.runtime_setting("time_limit", str(time_limit))
		self.sim.runtime_setting("simulation_realtimefactor", str(realtimefactor))
		self.sim.runtime_setting("environment", environment)
		self.sim.runtime_setting("policy", "./conf/state_action_matrices/exploration_policy_random.txt") # Use random policy
		self.robots = robots
		self.sim.run(robots) # Run it, and receive the fitness

	def save(self):
		self.H = fh.read_matrix(self.data_folder,"H")
		self.A = fh.read_matrix(self.data_folder,"A")
		self.E = fh.read_matrix(self.data_folder,"E")
		self.des = fh.read_matrix(self.data_folder,"des")
		np.savez(self.save_id+"_learning_data", des=self.des, H=self.H, A=self.A, E=self.E)
		print("Saved")

	def load(self,cmd):
		if np.size(cmd) == 1:
			file = fh.get_latest_file('data/*_learning_data.npz')
		else:
			file = cmd[1]
		data = np.load(file)
		self.H = data['H'].astype(float)
		self.E = data['E'].astype(float)
		self.A = data['A'].astype(float)
		self.des = data['des'].astype(float)
		self.save_id = file[0:file.find('_learning_data')]

	def optimize(self):
		p0 = np.ones([self.A.shape[1],int(self.A.max())]) / self.A.shape[1]
		self.result, self.policy, self.empty_states = opt.main(p0, self.des, self.H, self.A, self.E)
		print("Unknown states:" + str(self.empty_states))
		print('{:=^40}'.format(' Optimization '))
		print("Final fitness: " + str(self.result.fun))
		print("[ policy ]")
		np.set_printoptions(threshold=sys.maxsize)
		print(self.policy)
		np.savez(self.save_id+"_optimization", result=self.result, policy=self.policy, empty_states=self.empty_states)

	def disp(self):
		print(self.H)
		print(self.E)
		print(self.A)
		print("States desireability: ", str(self.des))
		
	def evaluate(self,robots=30,time_limit=100, realtimefactor=50,environment="square",runs=100):
		self.sim.make(clean=True, animation=False, logger=False, verbose=True)
		self.sim.runtime_setting("time_limit", str(time_limit))
		self.sim.runtime_setting("simulation_realtimefactor", str(realtimefactor))
		self.sim.runtime_setting("environment", environment)

		# Benchmark
		f_0 = []
		self.sim.runtime_setting("policy", "./conf/state_action_matrices/exploration_policy_random.txt") # Use random policy
		for i in range(0,runs):
			print('{:=^40}'.format(' Simulator run '))
			print("Run " + str(i) + "/" + str(runs))
			f_0 = np.append(f_0,self.sim.run(robots))

		# Optimize
		f_n = []
		policy_file = self.sim.path + "/conf/state_action_matrices/exploration_policy_evolved.txt"
		fh.save_to_txt(self.policy, policy_file)
		self.sim.runtime_setting("policy", policy_file) # Use random policy
		for i in range(0,runs):
			print('{:=^40}'.format(' Simulator run '))
			print("Run " + str(i) + "/"  +str(runs))
			f_n = np.append(f_n,self.sim.run(robots))
		np.savez(self.save_id+"_validation", f_0=f_0, f_n=f_n)

	def histplots(self,filename=None,show=True):
		# Load
		file = filename if filename is not None else self.save_id
		data_validation = np.load(file+"_validation.npz")
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
