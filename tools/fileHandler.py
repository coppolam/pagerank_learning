#!/usr/bin/env python3
"""
Collection of functions to handle files, load them, save them, ect
@author: Mario Coppola, 2020
"""
import numpy as np

def load_matrix(file):
    try:
        matrix = np.loadtxt(open(file, "rb"), delimiter=", \t", skiprows=1)
        return matrix
    except:
        raise ValueError("Matrix " + file + " could not be loaded! Exiting.")

def read_matrix(folder, name, file_format=".csv"):
    mat = load_matrix(folder + name + file_format)
    return mat
