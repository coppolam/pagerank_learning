#!/usr/bin/env python3
'''
Verify the behavior of the swarm according to the propositions

@author: Mario Coppola, 2020
'''

import networkx as nx
import numpy as np
from tools import graph

class verification():
	'''Verify the behavior of the swarm according to the propositions'''
	
	def __init__(self,H0,H1,E,policy,des):
		'''Initialize the verifier'''

		# Make directed graphs from adjacency matrices
		H0[H0>0] = 1.
		H1[H1>0] = 1.
		E[E>0] = 1.
		# print(H1)
		# print(E)

		self.GH0 = nx.from_numpy_matrix(H0.astype(int),create_using=nx.MultiDiGraph())
		self.GH = nx.from_numpy_matrix(H1.astype(int),create_using=nx.DiGraph())
		self.GE = nx.from_numpy_matrix(E.astype(int),create_using=nx.MultiDiGraph())
		# graph.print_graph(self.GH)
		
		# Ignore unknown nodes with no information
		# Get unknown nodes (d) and delete them
		d = list(nx.isolates(self.GH0))
		policy = np.delete(policy,d,axis=0)
		self.GH.remove_nodes_from(d)
		self.GE.remove_nodes_from(d)
		self.des = np.delete(des,d)
		
		# Extract observation sets
		self.static = np.argwhere(
			np.array(np.sum(policy,axis=1))<0.001).flatten() 
		self.active = np.argwhere(
			np.array(np.sum(policy,axis=1))>0.001).flatten() 
		self.desired = np.argwhere(np.array(self.des)>0.01).flatten()
		self.static_undesired = np.setdiff1d(self.static,self.desired)
		self.undesired = np.setdiff1d(self.GH.nodes,self.desired)
		self.active_undesired = np.setdiff1d(self.active,self.desired)

	def _check_to_all(self, G, set1, set2, find_all=True):
		'''
		Checks that in graph G, all nodes in set1 have a 
		directed path to all nodes in set2

		The find_all flag indicates whether you want to find
		all counterexamples or just the first one.
		'''
		# Initialize flag to False
		counterexample_flag = False
		
		# Foe each set check whether a link exists
		for s1 in set1:
			for s2 in set2:
				if nx.has_path(G,s1,s2) is False:
					print("(%i, %i), \n"%\
							(s1, s2),end = '')
					counterexample_flag = True

					# Find all counterexamples? 
					# If false, just stop here.
					if find_all is False:
						break

		# Return False if fail, True if passed
		return not counterexample_flag

	def _check_to_any(self, G, set1, set2):
		'''
		Checks that in graph G, all nodes in set1 have a directed 
		path to at least one node in set2
		'''
		# Initialize flag to False
		counterexample_flag = False
		
		for s1 in set1:
    			
    		# For a node, find a path to any s2 in set2
			any_flag = False
			for s2 in set2:
				if nx.has_path(G,s1,s2) is True: 
					any_flag = True
					break # Connection found
			
			# If you reach here, then no connection was found
			if any_flag is False:
				print("Counterexample found for node %i"%(s1))
				counterexample_flag = True
		
		# Return False if fail, True if passed
		return not counterexample_flag

	def _check_edge(self, G, set1, set2):
		'''
		Checks that in graph G, all nodes in set1 have a directed 
		path to at least one node in set2
		'''
		# Initialize flag to False
		counterexample_flag = False
		
		for s1 in set1:
    			
    		# For a node, find a path to any s2 in set2
			any_flag = False
			for s2 in set2:
				
				if G.has_edge(s1,s2) is True:		
					any_flag = True
					break # Connection found
			
			# If you reach here, then no connection was found
			if any_flag is False:
				print("Counterexample found for node %i"%(s1))
				counterexample_flag = True
		
		# Return False if fail, True if passed
		return not counterexample_flag


	def _condition_1(self):
		'''
		GS1 (H), shows that ALL desired states 
		can be reached from ALL states
		'''
		return self._check_to_all(self.GH,self.undesired,self.desired)

	def _condition_2(self):
		'''
		GS2 (E) shows that ALL static states that are not 
		desired can become active via the environment
		'''
		return self._check_edge(self.GE,self.static_undesired,self.active)

	def _condition_3(self):
		'''
		GS1 (H) shows that an active simplicial state can 
		transition "freely" to any other state
		'''
		return self._check_to_all(self.GH,self.active_undesired,self.GH.nodes)

	def disp(self):
		'''
		Print the set of static, active, and desired 
		observations to the terminal
		'''
		print("Static states:\n",self.static)
		print("Active states:\n",self.active)
		print("Desired states:\n",self.desired)

	def verify(self,verbose=True):
		'''Verify all conditions and return the result'''
		
		c = []

		if verbose: self.disp()
		
		if verbose: print("\nChecking Proposition 1")
		c.append(self._condition_1())

		if verbose: print("\nChecking Proposition 2, Condition 1")
		c.append(self._condition_2())
		
		if verbose: print("\nChecking Proposition 2, Condition 2")
		c.append(self._condition_3())
		
		print("\nResult:", c)

		return all(c)