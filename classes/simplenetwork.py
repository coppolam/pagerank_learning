#!/usr/bin/env python3
"""
Optimize a behavior based on the PageRank function
@author: Mario Coppola, 2020
"""
import torch

class simplenetwork:
	def __init__(self,D_in):
		self.network = self.initialize(D_in, 1000, 1)
		self.loss_fn = torch.nn.MSELoss(reduction='sum')
		self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4)
		# (Note: Adam optimizer works well, SGD unstablish with 1e-4 learning rate)

	def initialize(self, D_in, H, D_out):
		model = torch.nn.Sequential(
			torch.nn.Linear(D_in, H),
			torch.nn.ReLU (),
			torch.nn.Linear(H, D_out))
		return model

	def run(self,x,y):
		y_pred = self.network(x)
		loss = self.loss_fn(y_pred, y)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return y_pred, loss

	def get(self):
		return self.network
		
