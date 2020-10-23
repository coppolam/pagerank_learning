#!/usr/bin/env python3
"""
Curated matplotlib functions for easy pretty plotting for papers

@author: Mario Coppola, 2020
"""

# Load
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Set Latex stuff
# matplotlib.rc('text', usetex=True)
# plt.rc('text', usetex=True)

def setup(w=12,h=6,fs=30,font='serif'):
	'''
	Sets up a figure with a good size. 
	The default values are pretty good for papers.
	'''
	
	matplotlib.rcParams.update({'font.size': fs})
	plt.rc('font', family=font)
	plt.figure(figsize=(w,h))
	
	return plt

def adjust(plt):
	'''Adjusts parameters for better layout in a paper'''
	
	plt.gcf().subplots_adjust(bottom=0.15)
	
	return plt
