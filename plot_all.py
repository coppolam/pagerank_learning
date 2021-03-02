#!/usr/bin/env python3
'''
Script that generates all the plots
Useful if you are feeling lazy :)

@author: Mario Coppola, 2020
'''

import argparse,subprocess,sys

from plot import plot_nn
from plot import plot_model
from plot import plot_benchmark
from plot import plot_logs
from plot import plot_evolution
from plot import plot_onlinelearning
import main_verification as plot_verification
import main_nn_training as plot_prediction

def main(args):

    print("(Re-)generating all figures")

    # Parse the arguments
    parser = argparse.ArgumentParser(
            description='Make all the plots with one command')
    parser.add_argument('-format', type=str, default="pdf", 
                        help="(str) desired format, default=pdf")
    args = parser.parse_args(args)

    # Controller names
    controllers = ["aggregation",
                    "pfsm_exploration", 
                    "pfsm_exploration_mod", 
                    "forage"]
 
    # Make relevant plots for each controller
    for c in controllers:
        # ## Plot neural network prediction performance
        # ## (Figure 4)
        plot_nn.main([c,"-format",args.format])

        # ## Plot model convergence
        # ## (Figure 5)
        plot_model.main([c,"data/%s/training_data/"%c,"-format",args.format])
        
        # ## Plot neural network prediction performance over validation runs 
        # ##(Figure 6)
        plot_prediction.main(["data/%s/training_data/"%c,
                 "data/%s/validation_data_1/"%c,
                 "data/%s/"%c,
                 "-evaluate","-plot"])
        
        # ## Plot benchmarks
        # ## (Figure 7)
        plot_benchmark.main([c,"-format",args.format])
        
        # ## Plot logs of benchmark sims
        # ## (Figure 8)
        plot_logs.main([c,"data/%s/optimization_1/"%c,
                "data/%s/evolution/"%c,"-format",args.format])
        
        # ## Plot verification pagerank results
        # ## (Figure 9 + 10)
        plot_verification.main([c,"data/%s/optimization_1/"%c,"-plot",
                       "-format",args.format])
    
        # ## Plot evolutions
        # ## (Figure 11)
        plot_evolution.main(["data/%s/"%c])

        ## Plot onlinelearning
        ## (Figure 12)
        if c == "aggregation" or c == "pfsm_exploration":
            plot_onlinelearning.main([c,"-format",args.format])

    print("...done!")
    

if __name__ == "__main__":
    main(sys.argv[1:])
