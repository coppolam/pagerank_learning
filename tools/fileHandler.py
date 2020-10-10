#!/usr/bin/env python3
"""
Collection of functions to handle files, 
load them, save them, ect.

@author: Mario Coppola, 2020
"""

import os
import time
import glob
import pickle
import numpy as np

def load_matrix(file,delimiter=",",skiprows=0):
	'''Loads a matrix from a file and returns the matrix'''
	try:
		matrix = np.loadtxt(open(file, "rb"), 
			delimiter=delimiter, skiprows=skiprows)
		return matrix
	except:
		raise ValueError("Matrix " + file + 
				" could not be loaded! Exiting.")

def read_matrix(folder, name, file_format=".csv"):
	'''Reads a matrix file and returns the matrix'''
	# LOad a csv file
	mat = load_matrix(folder + name + file_format)
	return mat

def make_folder(folder):
	'''Generates a folder if it doesn not exist'''
	# Try to make a main folder 
	# (if it fails then the directory exists!)
	try:
		os.mkdir(folder)
	except:
		None
	
	# Make subfolder with the time of a simulation
	folder = folder + "/sim_" + time.strftime("%Y_%m_%d_%T")

	# Make it
	os.mkdir(folder)

	# Return the final folder name
	return folder + "/"

def save_to_txt(mat,name):
	'''Saves data to a txt file'''
	NEWLINE_SIZE_IN_BYTES = -1 # -2 on Windows?
	with open(name, 'wb') as fout:
		# Note 'wb' instead of 'w'
		np.savetxt(fout, mat, delimiter=" ", fmt='%.3f')
		fout.seek(NEWLINE_SIZE_IN_BYTES, 2)
		fout.truncate()

def get_latest_file(path):
	'''Gets the newest file in a specified path'''
	# Get a list of files from the path
	list_of_files = glob.glob(path)

	# Extract the one with the highest time
	return max(list_of_files, key=os.path.getctime)

def save_pkl(var,name):
	'''Stores a variable to a specified pkl file'''
	with open(name, "wb") as file:
		pickle.dump(var, file)

def load_pkl(name):
	'''Loads data from a specified pkl file'''
	# Open the file and load the pkl file
	with open(name, "rb") as file:
		data = pickle.load(file)
	return data

def clear_folder(_dir,_type="npz"):
	'''
	Clears all files in a folder for a specified format
	
	By default it will clear ALL files. Specify type=<format> to
	indicate the file format.
	'''
	files = [ f for f in os.listdir(_dir) if f.endswith("."+_type) ]
	for f in files:
		os.remove(os.path.join(_dir,f))
