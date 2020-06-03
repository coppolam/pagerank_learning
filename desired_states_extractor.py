import torch
import numpy as np
import simplenetwork
from tools import matrixOperations as matop
import evolution
import aggregation as env

class desired_states_extractor:
	def __init__(self):
		pass

	def make_model(self,x,y,plot=False):
		self.network = simplenetwork.simplenetwork(x.shape[1])
		i = 0
		loss_history = []
		for element in y:
			in_tensor = torch.tensor([x[i]]).float()
			out_tensor = torch.tensor([[element]]).float()
			_,loss = self.network.run(in_tensor,out_tensor)
			loss_history = np.append(loss_history,loss.item())
			i += 1
		return self.network, loss_history
	
	def evaluate_model(self,network,x,y,plot=False):
		y_pred = []
		for element in x:
			in_tensor = torch.tensor([element]).float()
			y_pred = np.append(y_pred,network(in_tensor).item())
		error = y_pred - y
		return error

	def extract_states(self,file):
		sim = env.aggregation()
		sim.load(file)
		local_states, fitness = sim.extract()
		self.dim = local_states.shape[1]
		return matop.normalize_rows(local_states), fitness
		
	def _fitness(self,individual):
		in_tensor = torch.tensor([individual]).float()
		f = self.network.network(in_tensor).item()
		return f, 

	def get_des(self):
		e = evolution.evolution()
		e.setup(self._fitness, GENOME_LENGTH=self.dim, POPULATION_SIZE=1000)
		p = e.evolve(verbose=False, generations=100)
		return e.get_best()

	def run(self,file,verbose=False):
		if verbose: print("Extracting data from log")
		s, f = self.extract_states(file)
		
		if verbose: print("Making the NN model")
		model = self.make_model(s, f)
		
		if verbose: print("Optimizing for desired states")
		des = self.get_des()
		
		if verbose: print("Desired states: " + str(des))
		
		return des