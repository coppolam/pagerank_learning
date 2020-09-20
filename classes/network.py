#!/usr/bin/env python3
"""
Optimize a behavior based on the PageRank function
@author: Mario Coppola, 2020
"""
import torch
# import net

class ffnetwork(torch.nn.Module):
	def __init__(self,n_inputs,n_outputs,layer_size=30,layers=3):
		'''Initialization function. Set here the hyperparameters'''
		super().__init__()
		
		# Layers
		self.fc_in = torch.nn.Linear(n_inputs,layer_size)
		self.fc_mid = torch.nn.Linear(layer_size,layer_size)
		self.fc_out = torch.nn.Linear(layer_size,n_outputs)

		# ReLU
		self.relu = torch.nn.ReLU()

		# Number of layers
		self.layers = layers

	def forward(self,x):
		# Run input layer
		x = self.relu(self.fc_in(x))

		# Run middle layers
		for i in range(self.layers):
			x = self.relu(self.fc_mid(x))

		# Output layer
		return self.fc_out(x)

class net:
	def __init__(self,n_inputs,n_outputs,layers,layer_size,lr=1e-5):
		'''Initialization function. Set here the hyperparameters'''
		self.network = ffnetwork(n_inputs,n_outputs,layer_size=30,layers=3)
		self.loss_fn = torch.nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

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