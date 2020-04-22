#!/usr/bin/env python3
"""
Simulate the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""

import aggregation, sys
rerun = False

sim = aggregation.aggregation()

if rerun:
	sim.run(time_limit=100)
	sim.save()
else:
	sim.load(sys.argv)
	sim.sim.plot_log()

# sim.disp()
# sim.optimize()
# sim.evaluate(runs=1)
# sim.histplots()