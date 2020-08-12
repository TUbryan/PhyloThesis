import networkx as nx
import random
import sys
from copy import deepcopy
from collections import deque
import re
import ast
import time
import numpy as np
from TMNetworkGenerator import *
import copy
from operator import itemgetter
import pickle
from TMAlgWithPairs import *

Data = np.load(r'C:\Users\bryan\Desktop\master thesis\PairData14JulWithCase3.npy', allow_pickle = True)

"""
Here we will find the features for the data that we are going to use to train a decision tree. This will
not be used to find the features when running the heuristic. For that a different implementation will be constructed.

1. The number of sub moves that need to be performed by the heuristic.
2. Distance between z and u in the  (u',z,u). This will be 0 if z is already a reticulation. We normalise this by the number of nodes that are not yet in the isomorphism.
3. Difference in depth of both u'and z in (u',z,u) if z is a reticulation, else we take the difference of depth between u'and u.
4/5. An indication if the node falls in case 1 or case 2. This will be two features with either 1 or 0.
6. Maybe we will implement the distance of the move.

"""


# the input node will look like this node = (u',z,u,case,network,target)
def SubMove(node):
    if node[3] == '1a' or node[3] == '2a' or node[3] == '3a':
        return 0
    if node[3] == '3bi' or node[3] == '3bii':
        return 1
    if node[3] == '1bi' or node[3] == '2bi' or node[3] == '3biiiA' or node[3] == '3biiiB':
        return 2
    if node[3] == '1biiA' or node[3] == '1biiB' or node[3] == '2biiA' or node[3] == '2biiB':
        return 3

def DistanceFinder(node):
    root1 = Root(node[4])
    root2 = Root(node[5])
    label_dict_1=Labels(node[4])
    label_dict_2=Labels(node[5])
    undir_network = node[4].to_undirected()
    undir_target = node[5].to_undirected()
    #initialize isomorphism
    isom_1_2 = dict()
    isom_2_1 = dict()
    isom_size = 0 
    for label,node1 in label_dict_1.items():
        node2=label_dict_2[label]
        isom_1_2[node1]=node2
        isom_2_1[node2]=node1
        isom_size+=1
        
        
    if node[3] == '1a' or node[3] == '1bi' or node[3] == '1biiA' or node[3] == '1biiB' or node[3] == '3a' or node[3] == '3bi' or node[3] == '3bii' or node[3] == '3biiiA' or node[3] == '3biiiB':
        if node[2] != None:
            Distance = nx.shortest_path_length(undir_network, node[1],node[2])
        else:
            Distance = 0
        return Distance / (len(undir_network.nodes())-len(isom_1_2))
    
    
    if node[3] == '2a' or node[3] == '2bi' or node[3] == '2biiA' or node[3] == '2biiB':
        if node[2] != None:
            Distance = nx.shortest_path_length(undir_target, node[1],node[2])
        else:
            Distance = 0
        return Distance / (len(undir_target.nodes())-len(isom_2_1))

def DifferenceDepthFinder(node):
    network = node[4]
    target = node[5]
    undir_network = network.to_undirected()
    undir_target = target.to_undirected()
    root_network = Root(network)
    root_target = Root(target)
    
    
    if node[3] == '1a' or node[3] == '1bi' or node[3] == '1biiA' or node[3] == '1biiB':
        if node[2] != None:
            Depth1 = nx.shortest_path_length(undir_target,root_target,node[0])
            Depth2 = nx.shortest_path_length(undir_network,root_network,node[2])
        else:
            Depth1 = nx.shortest_path_length(undir_target,root_target,node[0])
            Depth2 = nx.shortest_path_length(undir_network,root_network,node[1])
        return abs(Depth1 - Depth2)
    
    if node[3] == '2a' or node[3] == '2bi' or node[3] == '2biiA' or node[3] == '2biiB':
        if node[2] != None:
            Depth1 = nx.shortest_path_length(undir_network,root_network,node[0])
            Depth2 = nx.shortest_path_length(undir_target,root_target,node[2])
        else:
            Depth1 = nx.shortest_path_length(undir_network,root_network,node[0])
            Depth2 = nx.shortest_path_length(undir_target,root_target,node[1])
        return abs(Depth1 - Depth2)
    
    if node[3] == '3a':
        return 0
    if node[3] == '3bi' or node[3] == '3biiiA':
        Depth1 = nx.shortest_path_length(undir_target,root_target,node[0])
        Depth2 = nx.shortest_path_length(undir_network,root_network,node[1])
        return abs(Depth1-Depth2)
    if node[3] == '3bii' or node[3] == '3biiiB':
        Depth1 = nx.shortest_path_length(undir_target,root_target,node[0])
        Depth2 = nx.shortest_path_length(undir_network,root_network,node[2])
        return abs(Depth1-Depth2)
    
def Case1Determiner(node):
    if node[3] == '1a' or node[3] == '1bi' or node[3] == '1biiA' or node[3] == '1biiB':
        return 1
    else:
        return 0
    
def Case2Determiner(node):
    if node[3] == '2a' or node[3] == '2bi' or node[3] == '2biiA' or node[3] == '2biiB':
        return 1
    else:
        return 0
    
def Case3Determiner(node):
    if node[3] == '3a' or node[3] == '3bi' or node[3] == '3bii' or node[3] == '3biiiA' or node[3] == '3biiiB':
        return 1
    else:
        return 0
    
            
Data_List = []

for i in range(len(Data)):
    for j in range(len(Data[i])):
        feature1 = SubMove(Data[i][j][0])
        feature2 = DistanceFinder(Data[i][j][0])
        feature3 = DifferenceDepthFinder(Data[i][j][0])
        feature4 = Case1Determiner(Data[i][j][0])
        feature5 = Case2Determiner(Data[i][j][0])
        feature6 = Case3Determiner(Data[i][j][0])
        Data[i][j].append(feature1)
        Data[i][j].append(feature2)
        Data[i][j].append(feature3)
        Data[i][j].append(feature4)
        Data[i][j].append(feature5)
        Data[i][j].append(feature6)
        # Hier nog 0 en 1 laten appenden voor wanneer een keuze goed of slecht is 
        #en dan hebben we de trainingsdata in een format waar we mee kunnen werken!!
        if j <= 0.1*len(Data[i]):
            Data[i][j].append(1)
        else:
            Data[i][j].append(0)
        Data_List.append(Data[i][j])
    
    
    
np.save('FinalTest10aug',Data_List)    
    
    
    
    
    
    