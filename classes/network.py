#!/usr/bin/env python3
"""
Optimize a behavior based on the PageRank function
@author: Mario Coppola, 2020
"""
import torch

class ffnetwork(torch.nn.Module):
	def __init__(self,n_inputs,n_outputs,layer_size=30,layers=3):
		'''Initialization function. Set here the hyperparameters'''
		super().__init__()

		# Number of layers
		self.layers = layers
		
		# Define layers
		## Input, middle, and output layers
		self.fc_in = torch.nn.Linear(n_inputs,layer_size)
		self.fc_mid = torch.nn.Linear(layer_size,layer_size)
		self.fc_out = torch.nn.Linear(layer_size,n_outputs)

		## ReLU
		self.relu = torch.nn.ReLU()

	def forward(self,x):
		# Run input layer
		x = self.relu(self.fc_in(x))

		# Run middle layers
		for i in range(self.layers):
			x = self.relu(self.fc_mid(x))

		# Run output layer
		return self.fc_out(x)

class net:
	def __init__(self,n_inputs,n_outputs,layers,layer_size,lr=1e-5):
		'''Initialization function. Set here the hyperparameters'''
		# Define network
		self.network = ffnetwork(n_inputs,n_outputs,
						layer_size=layer_size,
						layers=layers)

		# MSE Loss function, to recreate desired outputs values
		self.loss_fn = torch.nn.MSELoss()

		# Adam optimizer
		self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

	def run(self, x, y):
		'''Runs the network with some new x and y data and optimizes it'''
		# Forward
		y_pred = self.network(x)

		# Loss
		loss = self.loss_fn(y_pred, y)

		# Backward and optimize
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return y_pred, loss

	def get(self):
		'''Returns the current network'''
		return self.network