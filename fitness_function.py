#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 15:03:19 2019

@author: mario
"""
import numpy as np

#a = np.array([[4, 6],[2, 6],[5, 2]])
#b = np.array([[1, 7],[1, 8],[2, 6],[2, 1],[2, 4],[4, 6],[4, 7],[5, 9],[5, 2],[5, 1]])
#aux.in1d_index(a,b)

def pagerank(G, tol=1e-6):
    # Iterative procedure
    n = G.shape[0] # Size of G
    pr = 1 / n * np.ones((1, n)) # Initialize Pagerank vector
    residual = 1 # Residual (initialize high, doesn't really matter)
    while residual >= tol:
        pr_previous = pr
        pr = np.matmul(pr,G) # Pagerank formula
        residual = np.linalg.norm(np.subtract(pr,pr_previous))
    return np.squeeze(np.asarray(pr))

def fitness_function(pr, s_des):
    f = np.mean(pr[s_des:])/np.mean(pr)
    return f

# TODO: Make
def alpha():
    # something something
    return alpha

# TODO: Make
def GSactive(Q,states):
    n_states = np.size(Q,0)
    n_choices = np.size(Q,1) - 1
    s = []
    t = []
    w = []
    for i in range(0,n_states):
        s_i = states[i,:]
        indexes = np.zeros([1,n_choices])
        for j in range(1,n_choices):
            indexes = aux.in1d_index([j s_i[2:]], states)
            # TODO: ismember??! return ordered list!
        np.append(s, i*np.ones([1,n_choices])]
        np.append(t, indexes)
        np.append(w, Q[i,:])
        
    Ga = gt.make_graph(s,t,w)
    return Ga
            
#nstates = size(states, 1);     # done
#bw = size(states, 2) - 1;      # done
#s = []; t = []; w = [];
#tgs1 = cell(1,nstates); 
#for i = 1:nstates #done
#    s_i = states(i,:);         # done
#    indexes = zeros(1,bw);     # done
#    for j = 1:bw               # done
#        [~, indexes(j)] = ismember([j s_i(2:end)],states,'rows'); #done
#    end
#    s = [s i*ones(size(indexes))]; # done
#    t = [t indexes];               # done
#    w = [w Q(i,:)];                # done
#    tgs1{i} = indexes;
#end
#gs1 = digraph(s,t,w);
#
#end
#

# TODO: Make
def GSpassive(states):
    n_states = np.size(Q,0)
    s = []
    t = []
    Gp = gt.make_graph(s,t)
    for i in range(0,n_states):
        s_i = states[i,:]   
        # Indexes with same zero sum
        indexes_samesum = 
        # Indexes with single change + not part of gs1
        indexes_single = 
        # Your own state stays the same
        indexes_single[states[indexes_single,0] ~= s_i[0]] = []
    np.append(s, i*np.ones([1,n_choices])]
    np.append(t, indexes)
    Gp = gt.make_graph(s,t)
    return Gp

#nstates = size(states, 1); #done
#s = []; #done
#t = []; #done
#tgs2 = cell(1,nstates); #noneed
#states = double(states); #noneed
#for i = 1:nstates #done
#    s_i = states(i,:); #done
#    % indexes with same zero sum
#    indexes_samesum = find(sum(s_i(2:end)) == sum(states(:,2:end),2))';
#    % indexes with single change + not part of gs1
#    indexes_single = indexes_samesum(max(abs(states(indexes_samesum,2:end)-s_i(2:end)),[],2)==1);
#    % your own state stays the same
#    indexes_single(states(indexes_single,1) ~= s_i(1)) = [];
#    
#    s = [s i*ones(size(indexes_single))]; #done
#    t = [t indexes_single]; #done
#    tgs2{i} = indexes_single; #noneed
#end
#gs2 = 0;%digraph(s,t); #done
    

def GoogleMatrix(alpha, H, E):
    M = np.size(H,0) # Matrix size
    alpha_mat = alpha   # aM = np.asmatrix(np.diag(alpha,8)) 
    Em = aux.normalize_rows(E) # Normalize environment matrix E
    
    # Get D for columns where H = 0
    #TODO: Fix this D thing
#    z = np.where(x <= 0.00001)[0] # Find transitions with no action
    D = np.matlib.zeros((M,M)) # Blank slate for matrix D 
#    D[z,:] = E[z,:] # Account for blocked states in matrix D
    
    # Normalize H
    S = aux.normalize_rows(H + D) # Normalize policy matrix S = H + D
    
    G = np.matmul(alpha_mat, S) + np.matmul(np.subtract(np.eye(alpha_mat.shape[0]), alpha_mat), Em)
    return G

def objF(x):
    Qtest[Qidx] = x;
    Ga = GSactive(Qtest,states)
    Gp = GSpassive(states)
    H = nx.adjacency_matrix(Ga)
    E = nx.adjacency_matrix(Gp)
    pr = pagerank(GoogleMatrix(alpha, H, E)) # Evaluate Pagerank vector 
    fitness = fitness_function(pr,s_des) # Get fitness
#    print(fitness) # DEBUG
    return fitness

#TODO: Make subclass of GA and add this there
def extract_history(l):
    # Extract fitness history
    fitness_history = []
    for x in range(0, l.numLearningSteps):
       fitness_history.append(max(l._allGenerations[x][1]))
    return fitness_history

def initialize_evolution_parameters(l):
    l.verbose = False # Verbose, defined on top
    l.maximize = True # Maximize the fitness function
    l.storeAllPopulations = True # Keep history
    l.populationSize = 10 # Population
    l.maxLearningSteps = 100 # Generations
    return l
