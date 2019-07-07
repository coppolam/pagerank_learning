#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 14:52:26 2019

@author: mario
"""
import numpy as np

def normalize_rows(mat):
    row_sums = np.squeeze(np.asarray(mat.sum(axis=1)))
    mat_norm = mat / row_sums[:, np.newaxis]
    return mat_norm