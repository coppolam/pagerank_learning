#!/usr/bin/env python3
"""
Loop the aggregation and optimize the behavior
@author: Mario Coppola, 2020
"""

import pickle, sys, matplotlib, os, argparse, copy
from tqdm import tqdm
import numpy as np
from classes import simulator, desired_states_extractor
from tools import fileHandler as fh
from tools import matrixOperations as matop
import matplotlib.pyplot as plt
matplotlib.rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('folder_training', type=str, help="(str) Training data folder")
parser.add_argument('folder_validation', type=str, help="(str) Validation data folder")
parser.add_argument('savefolder', type=str, help="(str) Save folder")
parser.add_argument('-train', type=bool, default=True, help="(bool) Train, defualt True")
parser.add_argument('-validate', type=bool, default=True, help="(bool) Validate, default True")
parser.add_argument('-debug', type=bool, default=False, help="(bool) Debug (only 10 iterations), default False")
args = parser.parse_args()

dse = desired_states_extractor.desired_states_extractor()
nets = []
filelist_training = [f for f in os.listdir(args.folder_training) if f.endswith('.npz')]
if args.debug: i = 0
if args.train:
	for filename in tqdm(sorted(filelist_training)):
		model = dse.train(args.folder_training+filename)
		nets.append(copy.deepcopy(model))
		if args.debug:
			i += 1
			if i > 20: break
	fh.save_pkl(nets,args.savefolder+"/models.pkl")
else:
	nets = fh.load_pkl(args.savefolder+"/models.pkl")

filelist_validation = [f for f in os.listdir(args.folder_validation+"/") if f.endswith('.npz')]
v = []
if args.debug: i = 0
if args.validate:
	for model in tqdm(nets):
		e = []
		for filename in sorted(filelist_validation): # Crosscheck against all  validatin files
			_, s, f = dse.extract_states(args.folder_validation+"/"+filename, pkl=True)
			_, corr, _ = dse.evaluate_model(model[0], s, f)
			e.append(np.mean(corr))
		v.append(e)
		if args.debug:
			i += 1
			if i > 20: break

vname = os.path.basename(os.path.dirname(args.folder_validation))
fh.save_pkl(v,args.savefolder + "/validation_" + vname + ".pkl")

# plt.plot(e)
# plt.show()

# # nets = fh.load_pkl(args.folder+"/models.pkl")
# # for i in range(len(nets)):

# # 	if i == 0: validationfile = file
# # 	# model = fh.load_pkl(file+"_model.pkl")
# # 	t,s,f = dse.extract_states(file,pkl=True)
# # 	# m[i] = np.trapz(f,x=t)/np.max(t)
# # 	for i in tqdm(range(5)): model,loss_history = dse.train_model(s, f) # Train the model
# # 	print("Loading validation set")
# # 	tc, sc, fc = dse.extract_states(validationfile)
# # 	print("Evaluating against validation set")
# # 	error, y_pred = dse.evaluate_model(model, sc, fc)
# # 	# Plot figure of error over time in validation dataset	
# # 	plt.plot(tc,fc);
# # 	plt.plot(tc,y_pred);
# # 	plt.xlabel("Time [s]")
# # 	plt.ylabel("Fitness [-]")
# # 	plt.gcf().subplots_adjust(bottom=0.15)
# # 	plt.draw()
# # 	plt.pause(0.001)
# # 	plt.clf()

# # des = des_nn.get_des()
# # policy = sim.optimize(policy, des)
# # print("Final")
# # print(policy)
# # sim.make(args.controller, agent, animation=True, verbose=False)
# # policy_filename = save_policy(sim, policy)
# # sim.run(time_limit=0, robots=args.n, environment="square", policy=policy_filename, 
# # 	pr_states=pr_states, pr_actions=pr_actions, run_id=args.id, fitness=fitness)
