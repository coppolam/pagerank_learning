#!/usr/bin/env python3
"""
Python API for swarmulator
@author: Mario Coppola, 2020
"""
import os, subprocess, threading, time, random, glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import concurrent.futures
from tools import fileHandler as fh

class swarmulator:
	'''Python API class for Swarmulator. Allows to interact with swarmulator via Python'''

	def __init__(self, path="../swarmulator",verbose=True):
		'''Load swarmulator object. Ensure that the correct path to swarmulator is given as input'''
		self.path = path
		self.verbose = verbose
		self._clear_pipes()

	def make(self, controller=None, agent=None, animation=False, logger=False, verbose=False, speed=True, clean=False):
		'''Builds swarmulator'''
		spd = " -j" if speed else ""
		ani = " ANIMATION=ON" if animation else ""
		log = " LOG=ON" if logger else ""
		vrb = " VERBOSE=ON" if verbose else ""
		ctrl = " CONTROLLER="+controller if controller else ""
		agnt = " AGENT="+agent if controller else ""
		if clean:
			subprocess.call("cd " + self.path + " && make clean ", shell=True)
		subprocess.call("cd " + self.path + " && make" + spd + ani + log + vrb + ctrl + agnt, shell=True)
		print("# Done")

	def _clear_pipes(self,folder="/tmp/"):
		fileList = glob.glob(folder+"swarmulator_*") # list of all pipes that have been created
		# Iterate over the list of filepaths and remove each file
		for filePath in fileList:
			try: 
				os.remove(filePath)
			except OSError:
				print("Error while deleting file")
    		
	def _launch(self, n, run_id):
		'''Launches an instance of a swarmulator simulation'''
		cmd = "cd " + self.path + " && ./swarmulator " + str(n) + " " + str(run_id) + " &"
		# print(cmd)
		subprocess.call(cmd, shell=True)
		if self.verbose: print("Launched instance of swarmulator with %s robots and pipe ID %s" % (n,run_id))

	def _get_fitness(self,pipe):
		'''Awaits to receive the fitness from a run'''
		while not os.path.lexists(pipe):
			time.sleep(0.0001) # wait for swarmulator to complete and create the pipe
		f = open(pipe).read() # FIFO pipe generated by swarmulator
		if self.verbose: print("Received fitness %s from pipe %s" % (str(f),pipe))
		return f

	def run(self, n, run_id=None):
		'''Runs swarmulator. If run_id is not specified, it will assign a random id'''
		self.run_id = random.randrange(10000000000) if run_id is None else run_id
		pipe = "/tmp/swarmulator_" + str(self.run_id)
		self._launch(n,run_id=self.run_id)
		f = self._get_fitness(pipe)
		return f

	def load(self,file=None):
		'''Loads the log of a swarmulator run. If id is not specified, it will take the most recent log'''
		if file is None:
			file = self.path + "/" + fh.get_latest_file(self.path + "/logs/log_*.txt")
			log = np.loadtxt(file)
		elif '.txt' in file:
			log = np.loadtxt(file)
		elif '.npz' in file:
			log = np.load(file)["log"].astype(float)
		else:
			raise ValueError("File format unknown!")
			return -1
		return log

	def plot_log(self, log=None, file=None, time_column=0, id_column=1, x_column=2, y_column=3):
		'''Visualizes the log of a swarmulator run'''
		if log is None:
			if file is None: log = self.load()
			else: log = self.load(file)
		robots = int(log[:,id_column].max())
		if self.verbose: print("Total number of robots: " + str(robots))
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		for x in range(1,robots):
			d = log[np.where(log[:,id_column] == x)]
			ax.plot(d[:,time_column],d[:,x_column],d[:,y_column])
		ax.set_xlabel("Time [s]")
		ax.set_ylabel("N [m]")
		ax.set_zlabel("E [m]")
		plt.show()

	def runtime_setting(self, setting, value):
		'''Assigns a value to a runtime setting of conf/parameters.xml'''
		s = "xmlstarlet edit -L -u \"/parameters/" + setting + "\" -v \""+ value + "\" " + self.path + "/conf/parameters.xml"
		subprocess.call(s, shell=True)
		if self.verbose: print("Runtime setting \"" + setting + "\" has been set to \"" + value + "\"")

	def get_runtime_setting(self, setting):
		'''Returns the value of a runtime parameter currently specified in swarmulator conf/parameters.xml'''
		s =  "xmlstarlet sel -t -v" +  " \"parameters/" +setting + "\" "+ self.path + "/conf/parameters.xml"
		value = subprocess.getoutput(s)
		if self.verbose: print("Runtime setting \"" + setting + "\" is \"" + value + "\"")
		return value;
	
	def batch_run(self,n,runs):
		'''Runs a batch of parallel simulations in parallel. By being different processes, the simulations can run unobstructed.'''
		self._clear_pipes()
		if isinstance(n,int): 
			robots = np.repeat(n,runs)
		elif len(n) == 2: 
			robots = np.random.randint(n[0],n[1],runs)
		
		if runs == 1:
			print("WARNING: Running simulator.batch_run but only using 1 run. Consider just using swarmulator.run instead")
		
		out = np.zeros(runs)
		c = 0
		with concurrent.futures.ProcessPoolExecutor() as executor:
			for i, f in zip(robots,executor.map(self.run, robots)):
				out[c] = float(f)
				c += 1
		
		return out
