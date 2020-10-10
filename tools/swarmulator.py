#!/usr/bin/env python3
"""
Python API for swarmulator
@author: Mario Coppola, 2020
"""

import os, subprocess, threading, time, random, glob
import numpy as np
import matplotlib.pyplot as plt # Plotting
import concurrent.futures # Needed for batch runs
from . import fileHandler as fh # Used to load/save files
from . import prettyplot as pp # Makes plots a little prettier

class swarmulator:
	''' Lower level Python API for Swarmulator.
	
		Allows to interact with swarmulator via Python
	'''

	def __init__(self, path="../swarmulator",verbose=True):
		''' Load swarmulator object.
			Ensure that the correct path to swarmulator is given as input
		'''
		
		# Set path to swarmulator
		self.path = path

		# Set whether this API will be in verbose mode or note
		self.verbose = verbose

		# Clear all current pipes
		self._clear_pipes()

	def make(self, controller=None, agent=None, speed=True, animation=False, 
					logger=False, verbose=False, clean=False):
		'''Build swarmulator'''

		# Set the build parameters
		spd = " -j" if speed else ""
		ani = " ANIMATION=ON" if animation else ""
		log = " LOG=ON" if logger else ""
		vrb = " VERBOSE=ON" if verbose else ""
		ctrl = " CONTROLLER=" + controller if controller else ""
		agnt = " AGENT=" + agent if controller else ""

		# Clean previous build
		if clean:
			subprocess.call("cd " + self.path + " && make clean ", shell=True)

		# Build
		subprocess.call("cd " + self.path + 
			" && make" + spd + ani + log + vrb + ctrl + agnt, 
			shell=True)

		print("# Done")

	def _get_log(self,file):
		'''
		Takes in a txt file logged from swarmulator and loads it.
		If no argument is given it uses the most recent log in the folder.
		'''
		if file is None:
			# Load the most recent file
			log = self.load()
		else:
			# Load the specified file
			log = self.load(file)

		# Return the log (numpy array)
		return log

	def _clear_pipes(self,folder="/tmp/"):
		'''Clear all current swarmulaotor FIFO pipes'''
			
		# Get a list of all relevant FIFO pipes that have been created
		fileList = glob.glob(folder+"swarmulator_*") 

		# Iterate over the list of filepaths and try to remove each file
		for filePath in fileList:
			try:
				os.remove(filePath)
			except OSError:
				print("Error while deleting file")

	def _launch(self, n, run_id):
		'''Launches an instance of a swarmulator simulation'''
		
		# Set up and launch the command
		cmd = "cd " + self.path \
					+ " && ./swarmulator " \
					+ str(n) + " " + str(run_id) + " &"
		subprocess.call(cmd, shell=True)

		# Write some info to the terminal
		if self.verbose:
			print("Launched instance of swarmulator with %s robots \
					and pipe ID %s" % (n,run_id))

	def _get_fitness(self,pipe):
		'''Awaits to receive the fitness from a run'''
		
		# Wait for swarmulator to complete and create the pipe
		while not os.path.lexists(pipe):
			time.sleep(0.0001)
		
		# Get fitness from pipe
		f = open(pipe).read()
		
		# Write some info to the terminal
		if self.verbose:
			print("Received fitness %s from pipe %s" % (str(f),pipe))

		# Return the received value, ensure float
		return float(f)

	def run(self, n, run_id=None):
		'''Runs swarmulator. If run_id is not specified, it will assign a random id'''
		
		# Give a random ID if not specified
		self.run_id = random.randrange(10000000000) if run_id is None else run_id

		# Set up a pipe object for the fitness to be communicated 
		# back to the API
		pipe = "/tmp/swarmulator_" + str(self.run_id)

		# Launch
		self._launch(n,run_id=self.run_id)

		# Wait for fitness from the pipe
		f = self._get_fitness(pipe)
		
		# Return the received value, ensure float
		return float(f)

	def load(self,file=None):
		'''Loads the log of a swarmulator run. 
		If id is not specified, it will take the most recent log'''
		
		# If the file is not specified, get the latest file
		if file is None:
			file = self.path + "/" + fh.get_latest_file(self.path + "/logs/log_*.txt")
			log = np.loadtxt(file)

		# If the file is a txt file, load it as txt
		elif '.txt' in file:
			log = np.loadtxt(file)

		# If the file is an npz file (assumed made by this API, then load the log)
		elif '.npz' in file:
			log = np.load(file)["log"].astype(float)

		# Something wrong, exit
		else:
			raise ValueError("File format unknown!")
			return -1
		
		return log

	def plot_log(self, log=None, file=None, time_column=0, 
						id_column=1, x_column=2, y_column=3, show=True):
		'''Visualizes the log of a swarmulator run'''
		if log is None:
			log = self._get_log(file)

		# Extract the number of robots
		robots = int(log[:,id_column].max())
		
		# Set up a 3D figure
		fig = plt.figure()
		ax = fig.gca(projection='3d')

		# Plot the path of each robot
		for x in range(1,robots+1):
			d = log[np.where(log[:,id_column] == x)]
			ax.plot(d[:,time_column],d[:,x_column],d[:,y_column])
		
		# Do some formatting, these values work quite well
		ax.set_xlabel("Time [s]") # Depth axis 3d plot
		ax.set_ylabel("N [m]") # Vertical axis 3d plot
		ax.set_zlabel("E [m]") # Horizontal axis 3d plot

		# Some pretty formatting that should work for many logs
		ax.xaxis.labelpad = pad
		ax.yaxis.labelpad = pad
		ax.zaxis.labelpad = pad
		ax.view_init(elev=36., azim=-38.)

		# Show the plot if desired
		if show is True:
			plt.show()

		# Return the plot object
		return plt

	def load_column(self, log=None, file=None, time_column=0, id_column=1, column=2):
		'''Visualizes the log of a swarmulator run'''

		if log is None:
			log = self._get_log(file)

		# Use the first robot as reference
		d = log[np.where(log[:,id_column] == 1)]

		# Return tuple with data
		return d[:,time_column], d[:,column]

	def plot_log_column(self, log=None, file=None, time_column=0, id_column=1, column=2, colname="parameter [-]", show=True, plot=None):
		'''Visualizes the log of a swarmulator run'''

		if log is None:
			log = self._get_log(file)

		# Initialize a new plot
		if plot is None:
			plt = pp.setup()
		else:
			plt = plot

		# Get data from first robot
		d = log[np.where(log[:,id_column] == 1)]
		
		# Plot the data
		plt.plot(d[:,time_column],d[:,column])
		plt.xlabel("Time [s]")
		plt.ylabel(colname)

		# Show
		if show is True:
			plt.show()

		return plt

	def runtime_setting(self, setting, value):
		'''Assigns a value to a runtime setting of conf/parameters.xml'''

		# Set up an xmlstarlet command to change the value
		s = "xmlstarlet edit -L -u \"/parameters/" \
			+ setting + "\" -v \"" \
			+ value + "\" "  \
			+ self.path + "/conf/parameters.xml"

		# Call the subprocess
		subprocess.call(s, shell=True)

		# Print
		if self.verbose:
			print("Runtime setting \"" + \
					setting + "\" has been set to \"" \
					+ value + "\"")

	def get_runtime_setting(self, setting):
		'''
		Returns the value of a runtime parameter currently 
		specified in swarmulator conf/parameters.xml
		'''

		# Set up an xmlstarlet command to get a value
		s = "xmlstarlet sel -t -v \"parameters/" \
			+ setting + "\" " \
			+ self.path + "/conf/parameters.xml"
		
		# Run the xmlstarlet call
		value = subprocess.getoutput(s)
		
		# Print
		if self.verbose:
			print("Runtime setting \"" + setting + "\" is \"" + value + "\"")
		
		return value;
	
	def batch_run(self,n,runs):
		'''
		Runs a batch of parallel simulations in parallel. 
		By being different processes, the simulations can run unobstructed.
		'''

		# Clear all the pipes for good housekeeping
		self._clear_pipes()

		# Get the number of robots, if n=scalar then it's known, 
		# if it's a list [ll, ul] use that
		if isinstance(n,int): 
			robots = np.repeat(n,runs)
		elif len(n) == 2: 
			robots = np.random.randint(n[0],n[1],runs)
		
		# Tell the user in case they are batching with only one runs
		if runs == 1:
			print("INFO: Running simulator.batch_run but only using 1 run.\
					Consider just using \"swarmulator.run\" instead")
		
		# Set up a fitness vector to store all outputs
		fitness_vector = np.zeros(runs)
		
		# Launch multiple threads and fill in the fitness_vector
		c = 0
		with concurrent.futures.ProcessPoolExecutor() as executor:
			for i, f in zip(robots,executor.map(self.run, robots)):
				out[c] = float(f)
				c += 1

		# Return array with fitness values
		return fitness_vector
