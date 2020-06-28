import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from classes import desired_states_extractor as dse
import numpy as np
from tools import fileHandler as fh
from tqdm import tqdm
dse = dse.desired_states_extractor()
n = 35
m = np.zeros(n)
fig = plt.figure()
import argparse
parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('file', type=str, help="(str) Relative path to npz log file used for analysis", default=None)
args = parser.parse_args()

for i in range(n):
	file = "data/"+args.file+"/loop_1/"+args.file+"_t500_r30_id1_%i.npz" % i
	if i == 0: validationfile = file
	# model = fh.load_pkl(file+"_model.pkl")
	t,s,f = dse.extract_states(file,pkl=True)
	# m[i] = np.trapz(f,x=t)/np.max(t)
	for i in tqdm(range(5)): model,loss_history = dse.train_model(s, f) # Train the model
	print("Loading validation set")
	tc, sc, fc = dse.extract_states(validationfile)
	print("Evaluating against validation set")
	error, y_pred = dse.evaluate_model(model, sc, fc)
	# Plot figure of error over time in validation dataset	
	plt.plot(tc,fc);
	plt.plot(tc,y_pred);
	plt.xlabel("Time [s]")
	plt.ylabel("Fitness [-]")
	plt.gcf().subplots_adjust(bottom=0.15)
	plt.draw()
	plt.pause(0.001)
	plt.clf()
	