#!/usr/bin/env python3
"""
Collection of curated matplotlib functions for easy pretty plotting
@author: Mario Coppola, 2020
"""
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def setup():
	plt.figure(figsize=(6,3))
	return plt

def adjust(plt):
	plt.gcf().subplots_adjust(bottom=0.15)
	return plt