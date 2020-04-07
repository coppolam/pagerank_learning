import subprocess
import threading

class swarmulator:
	def __init__(self, path):
		self.path = path
	
	def make(self, animation=False, logger=False, verbose=False, speed=True, clean=False):
		spd = " -j" if speed else ""
		ani = " ANIMATION=ON" if animation else ""
		log = " LOG=ON" if logger else ""
		vrb = " VERBOSE=ON" if verbose else ""
		if clean:
			subprocess.call("cd " + self.path + " && make clean ", shell=True)
		subprocess.call("cd " + self.path + " && make" + spd + ani + log + vrb, shell=True)

	def launch(self, n, i):
		subprocess.call("cd " + self.path + " && ./swarmulator " + str(n), shell=True)
		
	def get_fitness(self):
		f = open("/tmp/swarmulator").read() # FIFO pipe generated by swarmulator
		return f

	def run(self, n, i=0):
		t = threading.Thread(target=self.launch, args=(n,i,))
		t.start()
		f = self.get_fitness()
		return f
