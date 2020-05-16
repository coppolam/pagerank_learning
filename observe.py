import numpy as np
data = np.load("data/learning_data_particle_aggregation/benchmark.npz")

p_s = data["p_s"].astype(float)
p_n = data["p_n"].astype(float)


import aggregation as env
sim = env.aggregation()

print("Standard optim")
print(p_s)

print("new")
print(p_n)

log_s = sim.observe(p_s,time_limit=100,clean=True,controller="controller_aggregation",agent="particle")
log_n = sim.observe(p_n,time_limit=100,clean=False,controller="controller_aggregation",agent="particle")


sim.sim.plot_log(log=log_s)

sim.sim.plot_log(log=log_n)
