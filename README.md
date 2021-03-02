
### Get the data from the paper

You can download it automatically using the included script ```download_data.sh```.

This repository works with swarmulator branch ```SI_framework```: ```https://github.com/coppolam/SI_framework```

Note, the script ```download_simulator.sh``` will also download the appropriate swarmulator branch in zip format.

### Install Python required packages
```sudo pip3 install -r requirements.txt```

### Main scripts
These are all the "main" scripts that have been used to generate/process the data.

 - ```main_generate_data_training.py```: generates the training data for a specified study case to then train the micro-macro link and extract the transition model.

 - ```main_generate_data_benchmark.py```: generates performance data for random policies to see how random policies perform.

 - ```main_nn_training.py```: learns and evaluates the micro-macro link model

 - ```main_optimization.py```: optimizes a policy and benchmarks the results

 - ```main_verification.py```:verifies an optimized policy according to the conditions in propositions 1 and 2. Gives the data seen in Table 2 if supplied with the supplied data (see ```download_data.sh``` script)

 - ```main_evolve.py```: uses the DEAP package to do a "standard" simulation based evolution for a specified study case. To have it work properly, the ```evolution.setup()``` function may have to be altered with good mutation/mating parameters (by default, mutation is binary which is not useful for this case)

### Recreating the figures in the paper scripts
Use these scripts to create the figures used in the paper

#### Make all the figures in one go
 - ```plot_all.py```: generates all the plots in the paper, use with supplied data to recreate all figures exactly (see ```download_data.sh``` script in order to easily download the data)

#### Individual figure scripts (may require additional inputs, see plot_all.py)
 - ```plot/plot_nn.py```: recreates figures 4
 - ```plot/plot_model.py```: recreates figure 5
 - ```main_nn_training.py```: recreases figure 6 (but can also generate data!)
 - ```plot/plot_benchmark.py```: recreates figure 7 
 - ```plot/plot_logs.py```: recreates figure 8
 - ```main_verification.py```: recreates figures 9 and 10 (and also verifies propositions)
 - ```plot/plot_evolution.py```: recreates figure 11
 - ```plot/plot_onlinelearning.py```: recreates figure 12

### Packages
#### Classes
 - ```verification.py```: import and use the "get" function to get the conditions needed to run each study case properly

 - ```simplenetwork.py```: a higher level PyTorch class that sets up and runs a feedforward neural network. Change this class if you wish to change the parameters of a neural network

 - ```pagerank_optimization.py```: given a transition model and a set of desired observations, the "main" function in this class will optimize a policy and return the optimized policy according to Equation 6 the paper.

 - ```evolution.py```: a higher level DEAP API that sets up and runs the evolution. By default it has the settings to extract the desired observations.

 - ```simulator.py```: an API that can interface with swarmulator API at a higher level so that we can run and analyze all the simulations and logs easily

#### Tools
 - ```swarmulator.py```: This is a lower level Python API that can interface with the C++ simulator Swarmulator. The API can be used to set build and/or runtime parameters, build, read logs, start simulations, and more. A FIFO pipe is used by the API to read the fitness and the end of a run.
 - ```graph.py```: Graph functions built on top of the networkx package
 - ```fileHandler.py```: Functions to handle files, create folder, delete files, ect.
 - ```matrixOperations.py```: Functions for matrix handling, built on top of numpy
 - ```prettyplot.py```: Functions to help make prettier plots (for papers)
 - ```fitness_functions.py```: Re-implements fitness functions from swarmulator for post-processing, in case