# Script that generates all the plots
# Useful if you are feeling lazy :)

import argparse,subprocess

parser = argparse.ArgumentParser(description='Simulate a task to gather the data for optimization')
parser.add_argument('-format', type=str, default="pdf", help="(str) desired format, default=pdf")
args = parser.parse_args()

# PageRank plots / proofs
subprocess.call('python3 plot_paper_nn.py aggregation -format %s'%args.format, shell=True)
subprocess.call('python3 plot_paper_nn.py pfsm_exploration -format %s'%args.format, shell=True)
subprocess.call('python3 plot_paper_nn.py pfsm_exploration_mod -format %s'%args.format, shell=True)
subprocess.call('python3 plot_paper_nn.py forage -format %s'%args.format, shell=True)

# PageRank plots / proofs
subprocess.call('python3 plot_paper_benchmark.py aggregation -format %s'%args.format, shell=True)
subprocess.call('python3 plot_paper_benchmark.py pfsm_exploration -format %s'%args.format, shell=True)
subprocess.call('python3 plot_paper_benchmark.py pfsm_exploration_mod -format %s'%args.format, shell=True)
subprocess.call('python3 plot_paper_benchmark.py forage -format %s'%args.format, shell=True)

# PageRank plots / proofs
subprocess.call('python3 test_proof.py aggregation data/aggregation/training_data_1_1/ -plot -format %s'%args.format, shell=True)
subprocess.call('python3 test_proof.py pfsm_exploration data/pfsm_exploration/training_data_1_1/ -plot -format %s'%args.format, shell=True)
subprocess.call('python3 test_proof.py pfsm_exploration_mod data/pfsm_exploration_mod/training_data_1_1/ -plot -format %s'%args.format, shell=True)
subprocess.call('python3 test_proof.py forage data/forage/training_data_1_all/ -plot -format %s'%args.format, shell=True)

# Model plots
subprocess.call('python3 plot_paper_model.py aggregation data/aggregation/training_data_1_1/ -format %s'%args.format, shell=True)
subprocess.call('python3 plot_paper_model.py pfsm_exploration data/pfsm_exploration/training_data_1_1/ -format %s'%args.format, shell=True)
subprocess.call('python3 plot_paper_model.py pfsm_exploration_mod data/pfsm_exploration_mod/training_data_1_1/ -format %s'%args.format, shell=True)
subprocess.call('python3 plot_paper_model.py forage data/forage/training_data_1_1/ -format %s'%args.format, shell=True)
