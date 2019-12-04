This repository holds a sample Python (v3.7) code based on the paper:

**"The PageRank algorithm as a method to optimize swarm behavior through local analysis"**
*Mario Coppola, Jian Guo, Eberhard Gill, Guido de Croon, 2019.*
Swarm Intelligence, December 2019, Volume 13, Issue 3–4, pp 277–319

The paper is available open-access at this link: 
https://link.springer.com/article/10.1007/s11721-019-00172-z

PDF link:
https://link.springer.com/content/pdf/10.1007%2Fs11721-019-00172-z.pdf

The code was made for *Python 3.7*. 
Please ensure that the networkx library is installed in order to handle graphs.
Currently the optimization uses a relatively simple scipy implementation for demo purposes. This can be replaced by other optimizers too.

Run `main_consensus.py`.

# Troubleshooting notes
 * Make sure python 3.7 (specifically 3.7.5)
 * Make sure Networkx version is 2.4 (or later, albeit untested)
 * Configure your IDE (e.g., PyCharm) to use the correct python version
 * If you get an issue with PIL, then remember that Pillow is now the new PIL. Make sure there are no traces of PIL or else pillow will not work either.
 * Make sure you have python3.7-dev also installed, and the graphviz python package if you wish to also visualize the graphs.
