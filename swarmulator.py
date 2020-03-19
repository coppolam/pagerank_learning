import subprocess

class swarmulator:
    def __init__(self,path):
        self.path = path
    
    def make(self, animation=False, logger=False, speed=True):
        ani = " ANIMATION=ON" if animation else ""
        log = " LOGGER=ON" if logger else ""
        spd = " -j" if speed else ""
        subprocess.call("cd " + self.path + "&& make clean && make" + spd + ani + log, shell=True)

    def run(self,n,i):
        subprocess.call("cd " + self.path + " && mkdir hist" + str(i) + " && mv *.csv hist" + str(i) + "/", shell=True)
        subprocess.call("cd " + self.path + " && ./swarmulator " + str(n), shell=True)