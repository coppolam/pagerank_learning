import networkx as nx
import numpy as np

class verification():
	def __init__(self,H,E,policy,des):

		# Make graphs from adjacency matrices
		self.GH = nx.from_numpy_matrix(H)
		self.GE = nx.from_numpy_matrix(E)

		# Ignore unknown nodes with no information
		d = list(nx.isolates(self.GH))
		self.GH.remove_nodes_from(d)
		self.des = np.delete(des,d)
		a = 1 if policy.shape[1] > 1 else 0 # Axis
		policy = np.delete(policy,d,axis=a)

		# Extract states
		self.static = np.argwhere(np.array(np.sum(policy,axis=a))<0.001)
		self.active = np.argwhere(np.array(np.sum(policy,axis=a))>0.001)
		self.happy = np.argwhere(np.array(self.des)>0.1)
		self.static_unhappy = np.setdiff1d(self.static,self.happy)
   
	def _condition_1(self):
		'''GS1 (H), shows that all happy states can be reached'''
		counterexampleflag = False
		for node in range(len(self.GH.nodes)):
			for d in self.happy:
				if nx.has_path(self.GH,node,d[0]) is False:
					print("Counterexample found for path %i to %i"%(s, a[0]))
					counterexampleflag = True
		if counterexampleflag: return False
		return True


	def _condition_2(self):
		'''GS2 (E) shows that all static states that are not desired can become active via the environment'''
		counterexampleflag = False
		for s in self.static:
			for a in self.active:
				if nx.has_path(self.GE,s,a[0]) is False:
					print("Counterexample found for path %i to %i"%(s, a[0]))
					counterexampleflag = True
		if counterexampleflag: return False
		return True

	def _condition_3(self):
		'''GS1 (H) shows that an active simplicial state can transition "freely" to any other state'''
		counterexampleflag = False
		for s in self.active:
			for d in range(len(self.GH.nodes)):
				if nx.has_path(self.GH,s[0],d) is False:
					print("Counterexample found for path %i to %i"%(s[0], d))
					counterexampleflag = True
		if counterexampleflag: return False
		return True

	def disp(self):
		print("Static states:\n",self.static.T[0])
		print("Active states:\n",self.active.T[0])
		print("Happy states:\n",self.happy.T[0])
		print("Static and unhappy states:\n",self.static_unhappy.T)

	def verify(self):
		c = []
		print("Testing Condition 1"); c.append(self._condition_1())
		print("Testing Condition 2"); c.append(self._condition_2())
		print("Testing Condition 3"); c.append(self._condition_3())
		print("Result:", c)
		return all(c)