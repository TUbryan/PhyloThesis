"""
This program will generate network pairs with set reticulations and leaves.
This will be used to generate data that will be used for the tail move problem.
"""

import numpy as np
import networkx as nx
import random as rn
import ete3

def NetworkGenerator(n,k):
    #We first set limiting parameters
    N = nx.DiGraph()
    s = n+k-2
    kr = k
    N.add_node(0)
    N.add_edge(0,1)
    N.add_edge(1,2)
    N.add_edge(1,3)
    w=4
    #We will add either a reticulation or a tree node to the network
    while s >0 or kr >0:
        decision = np.random.choice(['ret','tree'],p=[kr/(s+kr),s/(s+kr)])
        if decision == 'ret':
            Leafset = [n for n in N.nodes() if N.out_degree(n) == 0]
#            M = []
#            for i in Leafset:
#                if (N.in_degree([n for n in N.predecessors(i)][0]) == 1 or N.in_degree([n for n in N.predecessors(i)][0]) == 0) and N.out_degree([n for n in N.predecessors(i)][0]) == 2:
#                    for j in N.successors([n for n in N.predecessors(i)][0]):
#                        if j == i:
#                            continue
#                        elif N.in_degree(j) == 1 and N.out_degree(j) == 2:
#                            M.append(i)
#                        elif N.out_degree(j) == 0 and N.in_degree(j) == 1:
#                           M.append(i)
##            M = list(dict.fromkeys(M))
            if (len(Leafset) == 1) or (len(Leafset) ==2 and [n for n in N.predecessors(Leafset[0])] == [n for n in N.predecessors(Leafset[1])]):
                u = rn.choice(Leafset)
                N.add_edge(u,w)
                w +=1
                N.add_edge(u,w)
                w +=1
                s = s-1
            else:
               sample = rn.sample(Leafset,2)
               while [n for n in N.predecessors(sample[0])] == [l for l in N.predecessors(sample[1])]:
                   sample = rn.sample(Leafset,2)
               u = sample[0]
               v = sample[1]
               pred = [n for n in N.predecessors(v)]
               N.add_edge(pred[0],u)
               N.remove_node(v)
               N.add_edge(u,w)
               w +=1
               kr = kr-1
        if decision == 'tree':
            Leafset = [n for n in N.nodes() if N.out_degree(n) == 0]
            u = rn.choice(Leafset)
            N.add_edge(u,w)
            w +=1
            N.add_edge(u,w)
            w +=1
            s = s-1
    return N


def PairGenerator(n,k): #Given a number of leaves and fixed reticulation number we compute a pair of networks.
    Network1 = NetworkGenerator(n,k)
    edges1 = list(Network1.edges())
    Network2 = NetworkGenerator(n,k)
#    Network1,Network2 = Relabeling(Network1,Network2,n)
    while list(Network2.edges()) == edges1:
        Network2 = NetworkGenerator(n,k)
    edges2 = list(Network2.edges())
    return (Network1, Network2)



def Relabeling(network1,network2,n):
    dict1 = {}
    dict2 = {}
    length = len(network1.nodes())
    internal_nodes = length-n
    internal_node_names = []
    leaf_node_names = []
    j = 0
    k = 0
    l = 0
    m = 0
    for i in range(0,internal_nodes):
        internal_node_names.append(i)
        
    for i in range(internal_nodes, length):
        leaf_node_names.append(i)
#    for node in network1.nodes():
#        if network1.out_degree(node) != 0:
#           dict1[node]=internal_node_names[j]
#           j += 1
#        else:
#            dict1[node] = leaf_node_names[k]
#            k += 1
#            
#    for node in network2.nodes():
#        if network2.out_degree(node) != 0:
#           dict2[node]= internal_node_names[l]
#           l += 1
#        else:
#            dict2[node] = leaf_node_names[m]
#            m += 1
#    network1 = nx.relabel_nodes(network1,dict1)
#    network2 = nx.relabel_nodes(network2,dict2)
    for node in network1:
        if network1.out_degree(node) != 0:
           network1.node[node]['label']= internal_node_names[j]
           j +=1
        else:
            network1.node[node]['label'] = leaf_node_names[k]
            k +=1
    for node in network2:
        if network2.out_degree(node) != 0:
           network2.node[node]['label']= internal_node_names[l]
           l +=1
        else:
           network2.node[node]['label'] = leaf_node_names[m]
           m +=1
    return (network1,network2)       
    
def ete_converter(Tree):
    root = 0
    subtrees = {node:ete3.Tree(name=node) for node in Tree.nodes()}
    [*map(lambda edge:subtrees[edge[0]].add_child(subtrees[edge[1]]), Tree.edges())]
    tree = subtrees[root]
    return tree
    


#For data gen we used 30 to 50 leaves and 5 to 10 reticulations   
#For first pair data generation we used 40 to 50 leaves and 19 to 22 reticulations
#7-7-2020 We will use 8 to 10 leaves and 3 to 4 reticulations for our new trainingdata.
#9-7-2020 Use 8 to 12 leaves and 3 to 5 reticulations.
#14-7-2020 Use 8 to 12 leaves and 3 to 5 reticulations.
#16-7-2020 Use 8 to 12 leaves and no retics to make tree node data.

def DataGenerator(Samples): #Generates the specified number of sample pairs
    DataSample = []
    for i in range(0,Samples):
        Leaves = rn.randint(8,12)
        Reticulation = rn.randint(3,5) #0
        Network = NetworkGenerator(Leaves,Reticulation)
        DataSample.append(Network)
    return DataSample
        
        
    










