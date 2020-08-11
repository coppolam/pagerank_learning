import networkx as nx
import numpy as np

class verification():
	def __init__(self,H0,H1,E,policy,des):

		# Make directed graphs from adjacency matrices
		self.GH0 = nx.from_numpy_matrix(H0,create_using=nx.MultiDiGraph())
		self.GH = nx.from_numpy_matrix(H1,create_using=nx.MultiDiGraph())
		self.GE = nx.from_numpy_matrix(E,create_using=nx.MultiDiGraph())
		# Ignore unknown nodes with no information
		d = list(nx.isolates(self.GH0))
		policy = np.delete(policy,d,axis=0)
		self.GH.remove_nodes_from(d)
		self.GE.remove_nodes_from(d)
		self.des = np.delete(des,d)

		# Extract states
		self.static = np.argwhere(np.array(np.sum(policy,axis=1))<0.001).flatten() 
		self.active = np.argwhere(np.array(np.sum(policy,axis=1))>0.001).flatten() 
		self.desired = np.argwhere(np.array(self.des)>0.1).flatten()
		self.static_undesired = np.setdiff1d(self.static,self.desired)

	def _check_to_all(self, G, set1, set2, find_all=True):
		'''Checks that in graph G, all nodes in set1 have a directed path to all nodes in set2'''
		counterexample_flag = False
		for s1 in set1:
			for s2 in set2:
				if nx.has_path(G,s1,s2) is False:
					# print("Counterexample found for path %i to %i"%(s1, s2))
					print("(%i,%i); "%(s1, s2),end = '')
					counterexample_flag = True
					if find_all is False: break
		if counterexample_flag: return False
		print("")
		return True

	def _check_to_any(self, G, set1, set2):
		'''Checks that in graph G, all nodes in set1 have a directed path to at least one node in set2'''
		counterexample_flag = False
		for s1 in set1:
			any_flag = False
			for s2 in set2:
				if nx.has_path(G,s1,s2) is True: 
					any_flag = True
					break # Connection found
			if any_flag is False:
				print("Counterexample found for node %i"%(s1))
				counterexample_flag = True
		if counterexample_flag: return False
		return True

	def _condition_1_strong(self):
		'''GS1 (H), shows that all desired states can be reached from all states'''
		return self._check_to_all(self.GH,range(len(self.GH.nodes)),self.desired)
	
	def _condition_1_weak(self):
		'''GS1 (H), hows that all desired states can be reached from all states'''
		return self._check_to_all(self.GH,self.static_undesired,self.desired)

	def _condition_2(self):
		'''GS2 (E) shows that all static states that are not desired can become active via the environment'''
		return self._check_to_any(self.GE,self.static,self.active)

	def _condition_3_strong(self):
		'''GS1 (H) shows that an active simplicial state can transition "freely" to any other state'''
		return self._check_to_all(self.GH,self.active,self.desired)

	def _condition_3_weak(self):
		'''GS1 (H) shows that an active simplicial state can transition "freely" to any other state'''
		return self._check_to_any(self.GH,self.active,self.desired)

	def disp(self):
		print("Static states:\n",self.static)
		print("Active states:\n",self.active)
		print("Desired states:\n",self.desired)

	def verify(self):
		c = []
		self.disp()
		print("Checking Proposition 1 (strong)"); c.append(self._condition_1_strong())
		print("Checking Proposition 1 (weak)"); c.append(self._condition_1_weak())
		print("Checking Proposition 2, Condition 1"); c.append(self._condition_2())
		print("Checking Proposition 2, Condition 2 (strong)"); c.append(self._condition_3_strong())
		print("Checking Proposition 2, Condition 2 (weak)"); c.append(self._condition_3_weak())
		print("Result:", c)
		return all(c)