
import argparse

class common_parser:
	def __init__(self):
		pass

	def initialize(self,description=None):
		self.parser = argparse.ArgumentParser(description=description)

		self.a = []

		self.a.append({'controller':"", 
			"type":str, 
			"help":"(str) Controller to use during evaluation"})
		
		# self.a.append({'other':"", 
		# 	"type":str, 
		# 	"default":"t",
		# 	"help":"(str) Controller to use during evaluation"})

		return self.a
	
	def run(self):
		print(self.a)
		for item in self.a:
			self.parser.add_argument(item)

		return parser.parse_args()