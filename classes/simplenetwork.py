#!/usr/bin/env python3
"""
Optimize a behavior based on the PageRank function
@author: Mario Coppola, 2020
"""
import torch

class simplenetwork:
	def __init__(self,D_in,lr=1e-5):
		'''Initialization function. Set here the hyperparameters'''
		self.network = self.initialize(D_in, 200, 1)
		self.loss_fn = torch.nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

	def initialize(self, D_in, H, D_out):
		'''Initializes the network. Adapt as you see fit.'''
		model = torch.nn.Sequential(
			torch.nn.Linear(D_in, H), torch.nn.ReLU (),
			torch.nn.Linear(H, H), torch.nn.ReLU (),
			torch.nn.Linear(H, H), torch.nn.ReLU (),
			torch.nn.Linear(H, H), torch.nn.ReLU (),
			torch.nn.Linear(H, H), torch.nn.ReLU (),
			torch.nn.Linear(H, D_out))
		return model

	def run(self, x, y):
		'''Runs the network with some new x and y data and optimizes it'''
		y_pred = self.network(x)
		loss = self.loss_fn(y_pred, y)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return y_pred, loss

	def get(self):
		'''Returns the current network'''
		return self.network
