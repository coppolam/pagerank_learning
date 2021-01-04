# Script that generates all the plots
# Useful if you are feeling lazy :)

import argparse,subprocess,sys

from plot import plot_nn
from plot import plot_model
from plot import plot_benchmark
from plot import plot_logs
from plot import plot_evolution
import main_verification as plot_v
import main_nn_training as plot_evo

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
        ## Plot neural network 
        plot_nn.main([c,"-format",args.format])

        plot_evo.main(["data/%s/training_data/"%c,
                 "data/%s/validation_data_1/"%c,
                 "data/%s/"%c,
                 "-evaluate","-plot"])

        ## Plot model
        plot_model.main([c,"data/%s/training_data/"%c,"-format",args.format])
        
        ## Plot benchmarks
        plot_benchmark.main([c,"-format",args.format])
        
        ## Plot logs of benchmark sims
        plot_logs.main([c,"data/%s/optimization_1/"%c,
                "data/%s/evolution/"%c,"-format",args.format])
        
        ## Plot evolutions
        plot_evolution.main(["data/%s/"%c])

        ## Plot verification pagerank results
        plot_v.main([c,"data/%s/optimization_1/"%c,"-plot",
                       "-format",args.format])
    
    print("...done!")
    

if __name__ == "__main__":
    main(sys.argv[1:])
