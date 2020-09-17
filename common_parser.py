
import argparse

class common_parser:
	def __init__(self):
		pass

	def initialize(self,description=None):
		self.parser = argparse.ArgumentParser (description=description)

		self.a = []

		self.a.append = {'controller':None, 
			"type":str, 
			"help":"(str) Controller to use during evaluation"}
		
		self.a.append = {'other':None, 
			"type":str, 
			"default":"t",
			"help":"(str) Controller to use during evaluation"}

		return self.a
	
	def run(self):
		for item in self.a:
			parser.add_argument(**item)

		return parser.parse_args()