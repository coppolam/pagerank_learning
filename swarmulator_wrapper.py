#!/usr/bin/env python3
"""
Example for python interface with swarmulator
@author: Mario Coppola, 2020
"""

from swarmulator import swarmulator

path = "../swarmulator"
robots = 20
s = swarmulator.swarmulator(path) # Init
s.make(clean=False,animation=True) # Build (if already built, you can skip this)
f = s.run(robots) # Run it, and receive the fitness.

print("Fitness received from Swarmulator pipe: " + str(f))
print("Done")