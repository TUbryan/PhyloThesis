import numpy as np
import networkx as nx
from RearrDistance_Tools import *
from TMNetworkGenerator import *
from TMTestEnvironment import *
import ete3
import copy

"""
In this file we find the features for the data and label the data with a 0 or a 1. Where 0 indicated a bad choice
and 1 indicates a good choice.

The features that are considered:
    - Common number of blobs with target network appeared/disappeared (disappeared will be negative)
    - Distance between the tails of edge that is moved and the edge that it is moved to.
    - The diffrence of the number of reticulations that are present in the correct blobs before and after the move.
    - Robinson-Foulds distance wrt the minimum spanning trees before and after a move would be performed.
    - Difference in depth of the head of the edge that we are going to move and the head of the edge that we move to. This will be divided by the maximum depth to take size of graphs into account
    - Number of common edges with the target after the move is performed minus the number of common edges before the move is performed.
"""



def Common_blob_change(network,target,move):
    undir_network = network.to_undirected()
    undir_target = target.to_undirected()
    current_blobs_network = list(nx.biconnected_components(undir_network))
    blobs_target = list(nx.biconnected_components(undir_target))
    tracker1 = 0
    for i in current_blobs_network:
        if i in blobs_target:
            tracker1+=1
    new_network = DoMove(network,move[0],move[1],move[2],check_validity=False)
    undir_new_network = new_network.to_undirected()
    blobs_new_network = list(nx.biconnected_components(undir_new_network))
    tracker2 = 0
    for i in blobs_new_network:
        if i in blobs_target:
            tracker2+=1
    Difference = tracker2 - tracker1
    return Difference

def Distance_getter(network,move):
    undir_network = network.to_undirected()
    shortest_path = nx.shortest_path_length(undir_network,move[1],move[2][0])
    longest_path = nx.dag_longest_path_length(network) #should we divide by the longest path?!
    return shortest_path/longest_path
    

def Blob_reticulation_comparison(network,target,move):
    undir_network = network.to_undirected()
    undir_target = target.to_undirected()
    new_network = DoMove(network,move[0],move[1],move[2],check_validity = False)
    undir_new_network = new_network.to_undirected()
    
    
    current_blobs_network = list(nx.biconnected_components(undir_network))
    blobs_network_after_move = list(nx.biconnected_components(undir_new_network))
    blobs_target = list(nx.biconnected_components(undir_target))
    
    
    reticulations_network = []
    reticulations_new_network = []
    reticulations_target = []
    
    
    for i in network.nodes():
        if network.in_degree(i) == 2:
            reticulations_network.append(i)
    for i  in new_network.nodes():
        if new_network.in_degree(i) ==2:
            reticulations_new_network.append(i)
    for i in target.nodes():
        if target.in_degree(i) == 2:
            reticulations_target.append(i)
            
            
    common_reticulations = list(set(reticulations_network).intersection(reticulations_target))
    common_reticulations_new = list(set(reticulations_new_network).intersection(reticulations_target))
    
    
    tracker1 = 0
    tracker2 = 0
    tracker3 = 0
    
    
    #Finds the number of blobs that have the correct reticulations with respect to the target network
    for i in common_reticulations:
        for j in current_blobs_network:
            if (i in j) and (j in blobs_target):
                tracker1+=1
    #Finds the number of blobs that have the correct reticulations with respect to the target network after the move is performed.
    for i in common_reticulations_new:
        for j in blobs_network_after_move:
            if (i in j) and (j in blobs_target):
                tracker2+=1
        
    #Finds the total number of blobs in the target network that hold reticulations.
    for i in reticulations_target:
        for j in blobs_target:
            if i in j:
                tracker3+=1
    return tracker2 - tracker1
    

def RF_Difference_Getter(network,target,move):               
    #We first find the minimum spanning tree for the current network and the target (Target could be something global for all moves)
    MST_current_network = nx.minimum_spanning_tree(network.to_undirected())
    MST_target_network = nx.minimum_spanning_tree(target.to_undirected())
    
    #Find the network after we perform the move.
    new_network = DoMove(network,move[0],move[1],move[2],check_validity=False)
    
    #Find the minimum spanning tree for the new network
    MST_new_network = nx.minimum_spanning_tree(new_network.to_undirected())
    
    #Now we convert the trees to an ete3 tree object to find the robinson_foulds distances.
    MST_current_network = ete_converter(MST_current_network)
    MST_target_network = ete_converter(MST_target_network)
    MST_new_network = ete_converter(MST_new_network)
    
    #Finding the robinson-foulds distances
    current_RF_distance = MST_target_network.robinson_foulds(MST_current_network)[0]
    new_RF_distance = MST_target_network.robinson_foulds(MST_new_network)[0]
    
    #Taking the difference: old - new 
    Difference = current_RF_distance - new_RF_distance
    
    return Difference
    
    
def Depth_difference_getter(network, move): 
    leaves = Leaves_getter(network)
    max_depth = max([nx.shortest_path_length(network,0,i) for i in leaves])
    
    #Determine the depth of the head of the edge that will be moved
    depth_head_moving_edge = nx.shortest_path_length(network,0,move[0][1])
    
    #Determine the depth of the head of the edge that is moved to.
    depth_head_receiving_edge = nx.shortest_path_length(network,0,move[2][1])
    
    #We take the absolute difference between the depths.
    Difference = abs(depth_head_moving_edge - depth_head_receiving_edge)
    return Difference/max_depth


def Edge_difference_getter(network,target,move):    
   new_network = DoMove(network,move[0],move[1],move[2],check_validity=False)
   tracker1 = 0
   tracker2 = 0
   for i in network.edges():
       if i in target.edges():
           tracker1 +=1
   for i in new_network.edges():
       if i in target.edges():
           tracker2 +=1
   return tracker2 - tracker1
    
def Leaves_getter(network):
    leaves = []
    for i in network.nodes():
        if network.out_degree(i) == 0:
            leaves.append(i)
    return leaves

def Data_Getter(n): #Here n is the number of networks that will be constructed.
    global tracertje
    tracertje = 0
    list_of_graphs = DataGenerator(n)
    Data_housing = []
    for i in range(0,len(list_of_graphs)):
        target = TargetNetwork(list_of_graphs[i],5)[0]
        Dataset = setup(list_of_graphs[i],target) 
        for j in Dataset:
           move_features = []
           move_features.append(j[0])
           feature1 = Common_blob_change(list_of_graphs[i],target,j[0])
           feature2 = Distance_getter(list_of_graphs[i],j[0])
           feature3 = Blob_reticulation_comparison(list_of_graphs[i],target,j[0])
           feature4 = RF_Difference_Getter(list_of_graphs[i],target,j[0])
           feature5 = Depth_difference_getter(list_of_graphs[i], j[0])
           feature6 = Edge_difference_getter(list_of_graphs[i],target,j[0])
           move_features.append(feature1)
           move_features.append(feature2)
           move_features.append(feature3)
           move_features.append(feature4)
           move_features.append(feature5)
           move_features.append(feature6)
           move_features.append(j[2])
           #This determines whether or not moves are good. This moves are ordered from shortest to longest and roughly the top 50% get qualified as good moves.
           if Dataset.index(j) <= len(Dataset)/2:
               move_features.append(1)
           else:
               move_features.append(0)
           Data_housing.append(move_features)
        tracertje +=1
        print(tracertje)
    return Data_housing



#Datarun = Data_Getter(15)
#np.save('TMData_with_new_feature',Datarun)
#    