


"""
In this file we will take a look at the heuristic proposed in Lemma 4.6.
In the Lemma an arbitrary node is selected in some cases. We are going to
look into the possibility to replace this with machine learning or for starters with
random choice.
"""

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

#Dec_Tree = pickle.load(open('TMPairDecTree7Jul.sav','rb'))
#Dec_Tree = pickle.load(open('TMPairDecTree8JulTest.sav','rb'))
#Dec_Tree = pickle.load(open('TMPairDecTree17JulWithCase3.sav','rb'))
#Dec_Tree = pickle.load(open('TMPairDecTree26JulWithCase3.sav','rb'))
Dec_Tree = pickle.load(open('TMPairDecTreeFinal10aug.sav','rb'))
#Dec_Tree2 = pickle.load(open('TMPairDecTree17JulTree.sav','rb'))

TMEXPDATA = np.load(r'C:\Users\bryan\Desktop\master thesis\TMEXPDATA.npy', allow_pickle = True)

# Contents
# - Move functions
# - Finding nodes in a network
# - Sequence finding Functions
# - Isomorphism
# - I/O Functions




################################################################################
################################################################################
################################################################################
########                                                           #############
########                       Move functions                      #############
########                                                           #############
################################################################################
################################################################################
################################################################################


#Checks whether an endpoint of an edge is movable.
def CheckMovable(network,moving_edge,moving_endpoint):
    if moving_endpoint == moving_edge[0]:
        #Tail move
        if network.in_degree(moving_endpoint) in (0,2):
            #cannot move the tail if it is a reticulation or root
            return False
    elif moving_endpoint == moving_edge[1]:
        #Head move
        if network.out_degree(moving_endpoint) in (0,2):
            #cannot move the head if it is a tree node or leaf
            return False
    else:
        #Moving endpoint is not part of the moving edge
        return False
    #Now check for triangles, by finding the other parent and child of the moving endpoint
    parent_of_moving_endpoint = Parent(network,moving_endpoint,exclude=[moving_edge[0]])
    child_of_moving_endpoint  =  Child(network,moving_endpoint,exclude=[moving_edge[1]])    
    #if there is an edge from the parent to the child, there is a triangle
    #Otherwise, it is a movable edge
    return not network.has_edge(parent_of_moving_endpoint,child_of_moving_endpoint)
   
   
   
#Checks whether the given move is valid.
def CheckValid(network, moving_edge, moving_endpoint, to_edge):
    if moving_edge == to_edge:
        return False
    if not CheckMovable(network,moving_edge,moving_endpoint):
        return False
    if moving_endpoint == moving_edge[0]:
        #tail move, check whether the to_edge is below the head of the moving edge
        if nx.has_path(network,moving_edge[1],to_edge[0]):
            #the move would create a cycle
            return False
        if to_edge[1]==moving_edge[1]:
            #the move would create parallel edges
            return False
    elif moving_endpoint == moving_edge[1]:
        #head move, check whether the to_edge is above the tail of the moving edge
        if nx.has_path(network,to_edge[1],moving_edge[0]):
            #the move would create a cycle
            return False
        if to_edge[0]==moving_edge[0]:
            #the move would create parallel edges
            return False
    else:
        #The moving endpoint is not part of the moving edge
        #Checked in CheckMovable as well, redundant?!
        return False
    #No parallel edges at start location
    #No cycle
    #No parallel edges at end location
    #So the move is valid
    return True
       
       
       
#Attempts to do the move of given moving_edge (with a given endpoint, the head or the tail), to the edge to_edge.
#if check_validity==False, the validity of the move will not be checked, and the move will be performed whether it is possible or not.
#To use this setting, make sure that the move is valid yourself!
#Returns False if the move is invalid
#Returns the new network if the move is valid.
def DoMove(network,moving_edge,moving_endpoint,to_edge,check_validity=True):
    #Check wether the move is valid if the user wants to.
    if check_validity:
        if not CheckValid(network,moving_edge,moving_endpoint,to_edge):
            return False
    from_edge = From_Edge(network,moving_edge,moving_endpoint)
    #Perform the move and return the result
    #If the move is to an edge adjacent to the moving endpoint, do nothing
    if moving_endpoint in to_edge:
        return network.copy()
    
    #else, do the actual moves.
    new_network = network.copy()
    new_network.remove_edges_from([(from_edge[0],moving_endpoint),(moving_endpoint,from_edge[1]),to_edge])
    new_network.add_edges_from([(to_edge[0],moving_endpoint),(moving_endpoint,to_edge[1]),from_edge])
    return new_network

    
    


#Returns all valid moves in the network
#List of moves in format (moving_edge,moving_endpoint,to_edge)
def AllValidMoves(network,tail_moves=True,head_moves=True):
    validMoves = []
    headOrTail = []
    if tail_moves:
        headOrTail.append(0)
    if head_moves:
        headOrTail.append(1)
    for moving_edge in network.edges():
        for to_edge in network.edges():
            for end in headOrTail:
                if CheckValid(network, moving_edge, moving_edge[end], to_edge):
                    validMoves.append((moving_edge, moving_edge[end], to_edge))
    return validMoves






    
   

    
    

################################################################################
################################################################################
################################################################################
########                                                           #############
########                  Finding nodes in a network               #############
########                                                           #############
################################################################################
################################################################################
################################################################################

#Returns all nodes below a given node (including the node itself)
def AllBelow(network,node):
    lengths = nx.single_source_shortest_path_length(network,node)
    return lengths.keys()
    
    
#Find the lowest nodes above a given set.
#The excluded set must include all leaves.
def LowestReticAndTreeNodeAbove(network,excludedSet,target, toss = 0):
    lowest_retic = []
    lowest_tree_node = []
    for node in network.nodes():
        if node not in excludedSet:
            for c in network.successors(node):
                if c not in excludedSet:
                    break
            #else runs if the loop was not ended by a break
            #this happens exactly when all of the children are in excludedSet
            else:
                if network.out_degree(node)==2:
                    lowest_tree_node.append(node)
                else:
                    lowest_retic.append(node)
#    if toss == 1:
#        try:
#           chosen_lowest_tree_node = None
#           for i in lowest_tree_node: 
#               prediction = Prediction1(i,network,target)
#               if prediction == 1:
#                   chosen_lowest_tree_node = i
#                   break
#               if chosen_lowest_tree_node == None:
#                   chosen_lowest_tree_node = random.choice(lowest_tree_node)
#        except:
#           chosen_lowest_tree_node = None
#        try:
#           chosen_lowest_retic = None
#           for i in lowest_retic: 
#               prediction = Prediction2(i,network,target)
#               if prediction == 1:
#                   chosen_lowest_retic = i
#                   break
#               if chosen_lowest_retic == None:
#                   chosen_lowest_retic = random.choice(lowest_retic)
#        except:
#           chosen_lowest_retic = None
#        return chosen_lowest_tree_node, chosen_lowest_retic
        
    try:
       chosen_lowest_tree_node = random.choice(lowest_tree_node)
    except:
        chosen_lowest_tree_node = None
    try:
       chosen_lowest_retic = random.choice(lowest_retic)
    except:
        chosen_lowest_retic = None
    return chosen_lowest_tree_node, chosen_lowest_retic

def LowestReticAndTreeNodeAboveList(network,excludedSet):
    lowest_retic = []
    lowest_tree_node = []
    for node in network.nodes():
        if node not in excludedSet:
            for c in network.successors(node):
                if c not in excludedSet:
                    break
            #else runs if the loop was not ended by a break
            #this happens exactly when all of the children are in excludedSet
            else:
                if network.out_degree(node)==2:
                    lowest_tree_node.append(node)
                else:
                    lowest_retic.append(node)
                    
    return lowest_retic, lowest_tree_node



#Find the highest nodes below a given set.
#The excluded set must include the root.
def HighestNodesBelow(network,excludedSet):
    highest_retic = None
    highest_tree_node = None
    highest_leaf = None
    for node in network.nodes():
        if node not in excludedSet:
            for c in network.predecessors(node):
                if c not in excludedSet:
                    break
            #else runs if the loop was not ended by a break
            #this happens exactly when all of the parents are in excludedSet
            else:
                if network.out_degree(node)==2:
                    highest_tree_node = node
                elif network.in_degree(node)==2:
                    highest_retic = node
                else:
                    highest_leaf = node
                if highest_retic!=None and highest_leaf!=None and highest_tree_node!=None:
                    #break if all types of highest nodes are found
                    break
    return highest_tree_node, highest_retic, highest_leaf


def FindTreeNode(network,excludedSet=[]):
    for node in network.nodes():
        if node not in excludedSet and network.out_degree(node) == 2 and network.in_degree(node) == 1:
            return node
    return None

def FindReticList(network,excludedSet=[]):
    Retics = []
    for node in network.nodes():
        if node not in excludedSet and network.in_degree(node) == 2:
            Retics.append(node)
    return Retics


def FindRetic(network,excludedSet=[]):
    Retics = []
    for node in network.nodes():
        if node not in excludedSet and network.in_degree(node) == 2:
            Retics.append(node)
    if Retics == []:
        return None
    else:
        chosen_retic = random.choice(Retics)
        return chosen_retic


def Parent(network,node,exclude=[],toss = 0):
    parents = []
    for p in network.predecessors(node):
        if p not in exclude:
            parents.append(p)
    if parents == []:
       return None
    else:
       chosen_parent = random.choice(parents)
       return chosen_parent
 
def Child(network,node,exclude=[]):
    for c in network.successors(node):
        if c not in exclude:
            return c
    return None
    
   
#Returns the root of a network
def Root(network):
    for node in network.nodes():
        if network.in_degree(node)==0:
            return node
    return None
    
#Returns a dictionary with node labels, keyed by the labels
def Labels(network):
    label_dict = dict()
    for node in network.nodes():
        node_label = network.node[node].get('label')
        if node_label:
            label_dict[node_label]=node
    return label_dict            


################################################################################
################################################################################
################################################################################
########                                                           #############
########                 Sequence finding Functions                #############
########                                                           #############
################################################################################
################################################################################
################################################################################

def Green_Line(network1,network2,head_moves=True):
    if head_moves:
        return Green_Line_rSPR(network1,network2) 
    return Green_Line_Tail(network1,network2)


   
   
def GL_Case1_Tail(N,Np,up,isom_N_Np,isom_Np_N):
    #use notation as in the paper
    #' is denoted p
    xp = Child(Np,up)
    x = isom_Np_N[xp]
    z = Parent(N,x,exclude=isom_N_Np.keys())
    #Case1a: z is a reticulation
    if N.in_degree(z)==2:
        return [],[],z,up
    #Case1b: z is not a reticulation
    #z is a tree node
    if CheckMovable(N,(z,x),z):
        #Case1bi: (z,x) is movable
        #Find a reticulation u in N not in the isomorphism yet
        #TODO: Can first check if the other parent of x suffices here, should heuristcally be better
        u = FindRetic(N,excludedSet=isom_N_Np.keys())
        v = Child(N,u)
        if v==x:
            return [],[],u,up
        #we may now assume v!=x
        if z==v:
            v = Child(N,z,exclude=[x])
            w = Parent(N,u)
            return [((z,v),z,(w,u))],[],u,up
        w = Parent(N,u,exclude=[z])
        return [((z,x),z,(u,v)),((z,v),z,(w,u))],[],u,up
    #Case1bii: (z,x) is not movable
    c = Parent(N,z)
    d = Child(N,z,exclude=[x])
    #TODO: b does not have to exist if we have an outdeg-2 root, this could be c!
    b = Parent(N,c)
    if N.in_degree(b)!=0:
        #Case1biiA: b is not the root of N
        a = Parent(N,b)
        #First do the move ((c,d),c,(a,b)), then Case1bi applies as (z,x) is now movable
        newN = DoMove(N,(c,d),c,(a,b))
        u = FindRetic(newN,excludedSet=isom_N_Np.keys())
        v = Child(newN,u)
        if v==x:
            #Note: this only happens if u=d, in which case we shouldn't even do the move, but the algorithm says to do it.
            #TODO: catch this by first checking both parents of x for being a reticulation not in the isomorphism yet
            return [((c,d),c,(a,b))],[],u,up
        #we may now assume v!=x
        if z==v:
            #This could happen if z==v and u==b
            v = Child(newN,z,exclude=[x])
            w = Parent(newN,u)
            return [((c,d),c,(a,b)),((z,v),z,(w,u))],[],u,up
        w = Parent(newN,u,exclude=[z])
        return [((c,d),c,(a,b)),((z,x),z,(u,v)),((z,v),z,(w,u))],[],u,up
    #Case1biiB: b is the root of N
    #Note: d is not in the isomorphism
    e = Child(N,d)
    if N.out_degree(x)==2:
        s = Child(N,x)
        t = Child(N,x,exclude=[s])
        if s==e:
            return [((x,t),x,(d,e))],[],d,up
        if t==e:
            return [((x,s),x,(d,e))],[],d,up
        return [((x,s),x,(d,e)),((x,e),x,(z,t)),((x,t),x,(d,s))],[],d,up
    if N.out_degree(e)==2:
        s = Child(N,e)
        t = Child(N,e,exclude=[s])
        if s==x:
            return [((e,t),e,(z,x))],[],d,up
        if t==x:
            return [((e,s),e,(z,x))],[],d,up
        return [((e,s),e,(z,x)),((e,x),e,(d,t)),((e,t),e,(z,s))],[],d,up
    #neither are tree nodes, so both must be leaves
    #In that case, there is no sequence between the two networks.    
    return [],[],False,False
    
   
def GL_Case3(N,Np,up,isom_N_Np,isom_Np_N):
    xp,yp = list(Np.successors(up))
    x     = isom_Np_N[xp]
    y     = isom_Np_N[yp]
    parents_x=set(N.predecessors(x))
    parents_y=set(N.predecessors(y))
    common_parents = parents_x & parents_y
    #Case3a: x and y have a common parent not in the isom
    for parent in common_parents:
        if parent not in isom_N_Np.keys():
            #then parent can be mapped to up
            return [],[],parent,up
    #Case3b: x and y do not have a common parent in the isomorphism
    #For both, find a parent not in the isomorphism yet
    #Both these parents are tree nodes, otherwise we would be in Case 1 or 2
    z_x = Parent(N,x,exclude=isom_N_Np.keys())
    z_y = Parent(N,y,exclude=isom_N_Np.keys())

    #Case3bi: (z_x,x) is movable                
    if CheckValid(N,(z_x,x),z_x,(z_y,y)):
        return [((z_x,x),z_x,(z_y,y))],[],z_x,up
    #Case3bii: (z_y,y) is movable
    if CheckValid(N,(z_y,y),z_y,(z_x,x)):
        return [((z_y,y),z_y,(z_x,x))],[],z_y,up
    #Case3biii: Neither (z_x,x) nor (z_y,y) is movable
    #i.e., both edges hang of the side of a triangle.
    #Find the top node of the triangle for x
    c_x = Parent(N,z_x)
    b_x = Parent(N,c_x)
    
    #Find the top node of the triangle for y
    c_y = Parent(N,z_y)
#    print()
#    print(z_y,N.edges())

    b_y = Parent(N,c_y)
    
    if N.in_degree(b_x)==0:
        #c_x is the child of the root
        #c_x!=c_y, so c_y is not the child of the root
        #swap the roles of x and y
        x  ,   y = y  ,   x
        z_x, z_y = z_y, z_x
        b_x, b_y = b_y, b_x
        c_x, c_y = c_y, c_x
    #c_x is not the child of the root
    #find a parent of b_x, and the bottom node of the triangle d_x
    a_x = Parent(N,b_x)
    d_x = Child(N,c_x,exclude=[x])
    return [((c_x,d_x),c_x,(a_x,b_x)),((z_x,x),z_x,(z_y,y))],[],z_x,up    
    
    



def Green_Line_Tail(network1,network2):
    #Find the root and labels of the networks
    root1 = Root(network1)
    root2 = Root(network2)
    label_dict_1=Labels(network1)
    label_dict_2=Labels(network2)

    #initialize isomorphism
    isom_1_2 = dict()
    isom_2_1 = dict()
    isom_size = 0 
    for label,node1 in label_dict_1.items():
        node2=label_dict_2[label]
        isom_1_2[node1]=node2
        isom_2_1[node2]=node1
        isom_size+=1


    #Keep track of the size of the isomorphism and the size it is at the end of the green line algorithm
    goal_size = len(network1)-1
        
    
    #init lists of sequence of moves
    #list of (moving_edge,moving_endpoint,from_edge,to_edge)
    seq_from_1 = []
    seq_from_2 = []
    #TODO keep track of lowest nodes?
    

    #Do the green line algorithm
    while(isom_size<goal_size):
        #Find lowest nodes above the isom in the networks:
        lowest_tree_node_network1, lowest_retic_network1 = LowestReticAndTreeNodeAbove(network1,isom_1_2.keys(),network2)
        lowest_tree_node_network2, lowest_retic_network2 = LowestReticAndTreeNodeAbove(network2,isom_2_1.keys(),network2)
        
        
        ######################################
        #Case1: a lowest retic in network1
        if lowest_retic_network1:
            #use notation as in the paper network1 = N', network2 = N, where ' is denoted p
            up = lowest_retic_network1
            moves_network_2,moves_network_1,added_node_network_2,added_node_network_1 = GL_Case1_Tail(network2,network1,up,isom_2_1,isom_1_2)
            if not added_node_network_1:
                return False
        ######################################
        #Case2: a lowest retic in network2
        elif lowest_retic_network2:
            #use notation as in the paper network2 = N', network1 = N, where ' is denoted p
            up = lowest_retic_network2
            moves_network_1,moves_network_2,added_node_network_1,added_node_network_2 = GL_Case1_Tail(network1,network2,up,isom_1_2,isom_2_1)
            if not added_node_network_1:
                return False        
        ######################################
        #Case3: a lowest tree node in network1
        else:
            #use notation as in the paper network1 = N, network2 = N'
            up    = lowest_tree_node_network2
            moves_network_1,moves_network_2,added_node_network_1,added_node_network_2 = GL_Case3(network1,network2,up,isom_1_2,isom_2_1)
        #Now perform the moves and update the isomorphism
        isom_1_2[added_node_network_1]=added_node_network_2
        isom_2_1[added_node_network_2]=added_node_network_1
        for move in moves_network_1:
            seq_from_1.append((move[0],move[1],From_Edge(network1,move[0],move[1]),move[2]))
            network1 = DoMove(network1,move[0],move[1],move[2],check_validity=False)
        for move in moves_network_2:
            seq_from_2.append((move[0],move[1],From_Edge(network2,move[0],move[1]),move[2]))
            network2 = DoMove(network2,move[0],move[1],move[2],check_validity=False)
        isom_size+=1
        
    #Add the root to the isomorphism, if it was there
    isom_1_2[root1]=root2
    isom_2_1[root2]=root1	
                    
    #invert seq_from_2, rename to node names of network1, and append to seq_from_1
    return seq_from_1 + ReplaceNodeNamesInMoveSequence(InvertMoveSequence(seq_from_2),isom_2_1)

def Green_Line_Tail_Prediction(network1,network2):
    #Find the root and labels of the networks
    root1 = Root(network1)
    root2 = Root(network2)
    label_dict_1=Labels(network1)
    label_dict_2=Labels(network2)

    #initialize isomorphism
    isom_1_2 = dict()
    isom_2_1 = dict()
    isom_size = 0 
    for label,node1 in label_dict_1.items():
        node2=label_dict_2[label]
        isom_1_2[node1]=node2
        isom_2_1[node2]=node1
        isom_size+=1


    #Keep track of the size of the isomorphism and the size it is at the end of the green line algorithm
    goal_size = len(network1)-1
        
    
    #init lists of sequence of moves
    #list of (moving_edge,moving_endpoint,from_edge,to_edge)
    seq_from_1 = []
    seq_from_2 = []
    #TODO keep track of lowest nodes?
    

    #Do the green line algorithm
    while(isom_size<goal_size):
        Possible_nodes = NodeConstructor(network1,network2,isom_1_2,isom_2_1)
        #Case 3, We find a lowest tree node in target network
        if len(Possible_nodes) >=2 and Possible_nodes[1] == 'tree':
            up = random.choice(Possible_nodes[0])
            moves_network_1,moves_network_2,added_node_network_1,added_node_network_2 = GL_Case3(network1,network2,up,isom_1_2,isom_2_1)
         
        #Predict a node
        else:
            Predicted_node = Prediction(Possible_nodes,network1,network2,isom_1_2,isom_2_1)
            #Case 1
            if Predicted_node[3] == '1a' or Predicted_node[3] == '1bi' or Predicted_node[3] == '1biiA' or Predicted_node[3] == '1biiB':
                up = Predicted_node[0]
                moves_network_1,moves_network_2,added_node_network_1,added_node_network_2 = GL_Case1_Tail_with_start(network1,network2,up,isom_1_2,isom_2_1,Predicted_node[1:3])
                if not added_node_network_1:
                    return False
            
            #Case 2
            if Predicted_node[3] == '2a' or Predicted_node[3] == '2bi' or Predicted_node[3] == '2biiA' or Predicted_node[3] == '2biiB':
                up = Predicted_node[0]
                moves_network_2,moves_network_1,added_node_network_2,added_node_network_1 = GL_Case1_Tail_with_start(network2,network1,up,isom_2_1,isom_1_2,Predicted_node[1:3])
                if not added_node_network_1:
                    return False
        
        #Now perform the moves and update the isomorphism
        isom_1_2[added_node_network_1]=added_node_network_2
        isom_2_1[added_node_network_2]=added_node_network_1
        for move in moves_network_1:
            seq_from_1.append((move[0],move[1],From_Edge(network1,move[0],move[1]),move[2]))
            network1 = DoMove(network1,move[0],move[1],move[2],check_validity=False)
        for move in moves_network_2:
            seq_from_2.append((move[0],move[1],From_Edge(network2,move[0],move[1]),move[2]))
            network2 = DoMove(network2,move[0],move[1],move[2],check_validity=False)
        isom_size+=1
        
    #Add the root to the isomorphism, if it was there
    isom_1_2[root1]=root2
    isom_2_1[root2]=root1	
                    
    #invert seq_from_2, rename to node names of network1, and append to seq_from_1
    return seq_from_1 + ReplaceNodeNamesInMoveSequence(InvertMoveSequence(seq_from_2),isom_2_1)

def NodeConstructor(network, target, isom1, isom2):
    List_of_pairs = []
    #First we find the lowest reticulations in target and in network.
    target_retic, target_tree_nodes = LowestReticAndTreeNodeAboveList(target,isom2)
    network_retic = LowestReticAndTreeNodeAboveList(network,isom1)[0]
    if target_retic == [] and network_retic == []:
        return (target_tree_nodes, 'tree')
    if target_retic != []:
        for up in target_retic:
            xp = Child(target,up)
            x = isom2[xp]
            z = Parent(network,x,exclude=isom1.keys())
            if network.in_degree(z)==2:
                List_of_pairs.append((up,z,None,'1a'))
            else:
                    if CheckMovable(network,(z,x),z):
                        possible_u = FindReticList(network,excludedSet=isom1.keys())
                        for i in possible_u:
                            List_of_pairs.append((up,z,i,'1bi'))
                    else:
                        c = Parent(network,z)
                        d = Child(network,z,exclude=[x])
                        #TODO: b does not have to exist if we have an outdeg-2 root, this could be c!
                        b = Parent(network,c)
                        if network.in_degree(b)!=0:
                            #Case1biiA: b is not the root of N
                            a = Parent(network,b)
                            #First do the move ((c,d),c,(a,b)), then Case1bi applies as (z,x) is now movable
                            newN = DoMove(network,(c,d),c,(a,b))
                            possible_u = FindReticList(newN,excludedSet=isom1.keys())
                            for i in possible_u:
                                List_of_pairs.append((up,z,i,'1biiA'))
                        if network.in_degree(b)==0:
                            List_of_pairs.append((up,z,None,'1biiB'))
    if network_retic != []:
        for up in network_retic:
            xp = Child(network,up)
            x = isom1[xp]
            z = Parent(target,x,exclude=isom2.keys())
            if target.in_degree(z)==2:
                List_of_pairs.append((up,z,None,'2a'))
            else:
                    if CheckMovable(target,(z,x),z):
                        possible_u = FindReticList(target,excludedSet=isom2.keys())
                        for i in possible_u:
                            List_of_pairs.append((up,z,i,'2bi'))
                    else:
                        c = Parent(target,z)
                        d = Child(target,z,exclude=[x])
                        #TODO: b does not have to exist if we have an outdeg-2 root, this could be c!
                        b = Parent(target,c)
                        if target.in_degree(b)!=0:
                            #Case1biiA: b is not the root of N
                            a = Parent(target,b)
                            #First do the move ((c,d),c,(a,b)), then Case1bi applies as (z,x) is now movable
                            newN = DoMove(target,(c,d),c,(a,b))
                            possible_u = FindReticList(newN,excludedSet=isom2.keys())
                            for i in possible_u:
                                List_of_pairs.append((up,z,i,'2biiA'))
                        if target.in_degree(b)==0:
                            List_of_pairs.append((up,z,None,'2biiB'))
    return List_of_pairs

def Green_Line_Tail_Prediction2(network1,network2):
    #Find the root and labels of the networks
    root1 = Root(network1)
    root2 = Root(network2)
    label_dict_1=Labels(network1)
    label_dict_2=Labels(network2)

    #initialize isomorphism
    isom_1_2 = dict()
    isom_2_1 = dict()
    isom_size = 0 
    for label,node1 in label_dict_1.items():
        node2=label_dict_2[label]
        isom_1_2[node1]=node2
        isom_2_1[node2]=node1
        isom_size+=1


    #Keep track of the size of the isomorphism and the size it is at the end of the green line algorithm
    goal_size = len(network1)-1
        
    
    #init lists of sequence of moves
    #list of (moving_edge,moving_endpoint,from_edge,to_edge)
    seq_from_1 = []
    seq_from_2 = []
    #TODO keep track of lowest nodes?
    

    #Do the green line algorithm
    while(isom_size<goal_size):
        network_retic_list = LowestReticAndTreeNodeAboveList(network1,isom_1_2)[0]
        target_retic_list = LowestReticAndTreeNodeAboveList(network2,isom_2_1)[0]
        
        Possible_nodes = NodeConstructor2(network1,network2,isom_1_2,isom_2_1)
#        print(Possible_nodes)
        #Predict a node
        if  network_retic_list == [] and  target_retic_list == []:
            Predicted_node = Prediction(Possible_nodes,network1,network2,isom_1_2,isom_2_1)
        else:
            Predicted_node = Prediction(Possible_nodes,network1,network2,isom_1_2,isom_2_1)
        
        #Case 1
        if Predicted_node[3] == '1a' or Predicted_node[3] == '1bi' or Predicted_node[3] == '1biiA' or Predicted_node[3] == '1biiB':
            up = Predicted_node[0]
            moves_network_1,moves_network_2,added_node_network_1,added_node_network_2 = GL_Case1_Tail_with_start(network1,network2,up,isom_1_2,isom_2_1,Predicted_node[1:3])
            if not added_node_network_1:
                return False
            
        #Case 2
        if Predicted_node[3] == '2a' or Predicted_node[3] == '2bi' or Predicted_node[3] == '2biiA' or Predicted_node[3] == '2biiB':
            up = Predicted_node[0]
            moves_network_2,moves_network_1,added_node_network_2,added_node_network_1 = GL_Case1_Tail_with_start(network2,network1,up,isom_2_1,isom_1_2,Predicted_node[1:3])
            if not added_node_network_1:
                return False
        #Case 3, We find a lowest tree node in target network
        if Predicted_node[3] == '3a' or Predicted_node[3] == '3bi' or Predicted_node[3] == '3bii' or Predicted_node[3] == '3biiiA' or Predicted_node[3] == '3biiiB':
            up = Predicted_node[0]
            moves_network_1,moves_network_2,added_node_network_1,added_node_network_2 = GL_Case3_with_start(network1,network2,up,isom_1_2,isom_2_1,Predicted_node[1:3])
         
        #Now perform the moves and update the isomorphism
        isom_1_2[added_node_network_1]=added_node_network_2
        isom_2_1[added_node_network_2]=added_node_network_1
        for move in moves_network_1:
            seq_from_1.append((move[0],move[1],From_Edge(network1,move[0],move[1]),move[2]))
            network1 = DoMove(network1,move[0],move[1],move[2],check_validity=False)
        for move in moves_network_2:
            seq_from_2.append((move[0],move[1],From_Edge(network2,move[0],move[1]),move[2]))
            network2 = DoMove(network2,move[0],move[1],move[2],check_validity=False)
        isom_size+=1
        
    #Add the root to the isomorphism, if it was there
    isom_1_2[root1]=root2
    isom_2_1[root2]=root1	
                    
    #invert seq_from_2, rename to node names of network1, and append to seq_from_1
    return seq_from_1 + ReplaceNodeNamesInMoveSequence(InvertMoveSequence(seq_from_2),isom_2_1)

def NodeConstructor2(network, target, isom1, isom2):
    List_of_pairs = []
    #First we find the lowest reticulations in target and in network.
    target_retic, target_tree_nodes = LowestReticAndTreeNodeAboveList(target,isom2)
    network_retic = LowestReticAndTreeNodeAboveList(network,isom1)[0]
    if target_tree_nodes != []:
        for up in target_tree_nodes:
            xp,yp = list(target.successors(up))
            x     = isom2[xp]
            y     = isom2[yp]
            parents_x=set(network.predecessors(x))
            parents_y=set(network.predecessors(y))
            common_parents = parents_x & parents_y
            #Case3a: x and y have a common parent not in the isom
            for parent in common_parents:
                if parent not in isom1.keys():
                    List_of_pairs.append((up,None,None,'3a'))
            if list(common_parents) == []:
               z_x = Parent(network,x,exclude=isom1.keys())
               z_y = Parent(network,y,exclude=isom1.keys())
               
               #Case3bi: (z_x,x) is movable                
               if CheckValid(network,(z_x,x),z_x,(z_y,y)):
                   List_of_pairs.append((up,z_x,z_y,'3bi'))
               #Case3bii: (z_y,y) is movable
               if CheckValid(network,(z_y,y),z_y,(z_x,x)):
                   List_of_pairs.append((up,z_x,z_y,'3bii'))
               elif target_retic == [] and network_retic == []:
                   c_x = Parent(network,z_x)
                   b_x = Parent(network,c_x)
                   
                   #Find the top node of the triangle for y
                   c_y = Parent(network,z_y)
                   #    print()
                   #    print(z_y,N.edges())
                   
                   b_y = Parent(network,c_y)
                   
                   if network.in_degree(b_x)==0:
                       List_of_pairs.append((up,z_x,z_y,'3biiiB'))
                   else:
                       List_of_pairs.append((up,z_x,z_y,'3biiiA'))
                   
               
    if target_retic != []:
        for up in target_retic:
            xp = Child(target,up)
            x = isom2[xp]
            z = Parent(network,x,exclude=isom1.keys())
            if network.in_degree(z)==2:
                List_of_pairs.append((up,z,None,'1a'))
            else:
                    if CheckMovable(network,(z,x),z):
                        possible_u = FindReticList(network,excludedSet=isom1.keys())
                        for i in possible_u:
                            List_of_pairs.append((up,z,i,'1bi'))
                    else:
                        c = Parent(network,z)
                        d = Child(network,z,exclude=[x])
                        #TODO: b does not have to exist if we have an outdeg-2 root, this could be c!
                        b = Parent(network,c)
                        if network.in_degree(b)!=0:
                            #Case1biiA: b is not the root of N
                            a = Parent(network,b)
                            #First do the move ((c,d),c,(a,b)), then Case1bi applies as (z,x) is now movable
                            newN = DoMove(network,(c,d),c,(a,b))
                            possible_u = FindReticList(newN,excludedSet=isom1.keys())
                            for i in possible_u:
                                List_of_pairs.append((up,z,i,'1biiA'))
                        if network.in_degree(b)==0:
                            List_of_pairs.append((up,z,None,'1biiB'))
    if network_retic != []:
        for up in network_retic:
            xp = Child(network,up)
            x = isom1[xp]
            z = Parent(target,x,exclude=isom2.keys())
            if target.in_degree(z)==2:
                List_of_pairs.append((up,z,None,'2a'))
            else:
                    if CheckMovable(target,(z,x),z):
                        possible_u = FindReticList(target,excludedSet=isom2.keys())
                        for i in possible_u:
                            List_of_pairs.append((up,z,i,'2bi'))
                    else:
                        c = Parent(target,z)
                        d = Child(target,z,exclude=[x])
                        #TODO: b does not have to exist if we have an outdeg-2 root, this could be c!
                        b = Parent(target,c)
                        if target.in_degree(b)!=0:
                            #Case1biiA: b is not the root of N
                            a = Parent(target,b)
                            #First do the move ((c,d),c,(a,b)), then Case1bi applies as (z,x) is now movable
                            newN = DoMove(target,(c,d),c,(a,b))
                            possible_u = FindReticList(newN,excludedSet=isom2.keys())
                            for i in possible_u:
                                List_of_pairs.append((up,z,i,'2biiA'))
                        if target.in_degree(b)==0:
                            List_of_pairs.append((up,z,None,'2biiB'))
    return List_of_pairs

def GL_Case3_with_start(N,Np,up,isom_N_Np,isom_Np_N,nodes):
    xp,yp = list(Np.successors(up))
    x     = isom_Np_N[xp]
    y     = isom_Np_N[yp]
    parents_x=set(N.predecessors(x))
    parents_y=set(N.predecessors(y))
    common_parents = parents_x & parents_y
    #Case3a: x and y have a common parent not in the isom
    for parent in common_parents:
        if parent not in isom_N_Np.keys():
            #then parent can be mapped to up
            return [],[],parent,up
    #Case3b: x and y do not have a common parent in the isomorphism
    #For both, find a parent not in the isomorphism yet
    #Both these parents are tree nodes, otherwise we would be in Case 1 or 2
    z_x = nodes[0]
    z_y = nodes[1]

    #Case3bi: (z_x,x) is movable                
    if CheckValid(N,(z_x,x),z_x,(z_y,y)):
        return [((z_x,x),z_x,(z_y,y))],[],z_x,up
    #Case3bii: (z_y,y) is movable
    if CheckValid(N,(z_y,y),z_y,(z_x,x)):
        return [((z_y,y),z_y,(z_x,x))],[],z_y,up
    #Case3biii: Neither (z_x,x) nor (z_y,y) is movable
    #i.e., both edges hang of the side of a triangle.
    #Find the top node of the triangle for x
    c_x = Parent(N,z_x)
    b_x = Parent(N,c_x)
    
    #Find the top node of the triangle for y
    c_y = Parent(N,z_y)
#    print()
#    print(z_y,N.edges())

    b_y = Parent(N,c_y)
    
    if N.in_degree(b_x)==0:
        #c_x is the child of the root
        #c_x!=c_y, so c_y is not the child of the root
        #swap the roles of x and y
        x  ,   y = y  ,   x
        z_x, z_y = z_y, z_x
        b_x, b_y = b_y, b_x
        c_x, c_y = c_y, c_x
    #c_x is not the child of the root
    #find a parent of b_x, and the bottom node of the triangle d_x
    a_x = Parent(N,b_x)
    d_x = Child(N,c_x,exclude=[x])
    return [((c_x,d_x),c_x,(a_x,b_x)),((z_x,x),z_x,(z_y,y))],[],z_x,up 




    
def Prediction(nodes,network,target,isom1,isom2):
    tracker_nodes = copy.deepcopy(nodes)
    prediction = False
    while prediction == False:
        if tracker_nodes == []:
            return random.choice(nodes)
        feature_list = []
        node = random.choice(tracker_nodes)
        feat1 = SubMove(node)
        feat2 = DistanceFinder(node,network,target,isom1,isom2)
        feat3 = DifferenceDepthFinder(node,network,target)
        feat4 = Case1Determiner(node)
        feat5 = Case2Determiner(node)
        feat6 = Case3Determiner(node)
        feature_list.append(feat1)
        feature_list.append(feat2)
        feature_list.append(feat3)
        feature_list.append(feat4)
        feature_list.append(feat5)
        feature_list.append(feat6)
        pred = Dec_Tree.predict((np.array(feature_list).reshape(1,-1)))
        tracker_nodes.remove(node)
        if pred ==1:
            prediction == True
    return node

def Prediction2(nodes,network,target,isom1,isom2):
    tracker_nodes = copy.deepcopy(nodes)
    prediction = False
    while prediction == False:
        if tracker_nodes == []:
            return random.choice(nodes)
        feature_list = []
        node = random.choice(tracker_nodes)
        feat1 = SubMove(node)
        feat2 = DistanceFinder(node,network,target,isom1,isom2)
        feat3 = DifferenceDepthFinder(node,network,target)
#        feat4 = Case1Determiner(node)
#        feat5 = Case2Determiner(node)
#        feat6 = Case3Determiner(node)
        feature_list.append(feat1)
        feature_list.append(feat2)
        feature_list.append(feat3)
#        feature_list.append(feat4)
#        feature_list.append(feat5)
#        feature_list.append(feat6)
        pred = Dec_Tree2.predict((np.array(feature_list).reshape(1,-1)))
        tracker_nodes.remove(node)
        if pred ==1:
            prediction == True
    return node
        


#The next five functions are used to generate the features        
                           
def SubMove(node):
    if node[3] == '1a' or node[3] == '2a' or node[3] == '3a':
        return 0
    if node[3] == '3bi' or node[3] == '3bii':
        return 1
    if node[3] == '1bi' or node[3] == '2bi' or node[3] == '3biiiA' or node[3] == '3biiiB':
        return 2
    if node[3] == '1biiA' or node[3] == '1biiB' or node[3] == '2biiA' or node[3] == '2biiB':
        return 3

def DistanceFinder(node,network,target,isom1,isom2):
    undir_network = network.to_undirected()
    undir_target = target.to_undirected()
    #initialize isomorphism
    isom_1_2 = isom1
    isom_2_1 = isom2
    if node[3] == '1a' or node[3] == '1bi' or node[3] == '1biiA' or node[3] == '1biiB':
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
    
    
    if node[3] == '3a' or node[3] == '3bi' or node[3] == '3bii' or node[3] == '3biiiA' or node[3] == '3biiiB':
        if node[2] != None:
            Distance = nx.shortest_path_length(undir_network, node[1],node[2])
        else:
            Distance = 0
        return Distance / (len(undir_network.nodes())-len(isom_1_2))

def DifferenceDepthFinder(node,network,target):
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
        
        




def Green_Line_Tail_start_node(network1,network2,start_nodes):
    #Find the root and labels of the networks
    root1 = Root(network1)
    root2 = Root(network2)
    label_dict_1=Labels(network1)
    label_dict_2=Labels(network2)

    #initialize isomorphism
    isom_1_2 = dict()
    isom_2_1 = dict()
    isom_size = 0 
    for label,node1 in label_dict_1.items():
        node2=label_dict_2[label]
        isom_1_2[node1]=node2
        isom_2_1[node2]=node1
        isom_size+=1


    #Keep track of the size of the isomorphism and the size it is at the end of the green line algorithm
    goal_size = len(network1)-1
        
    
    #init lists of sequence of moves
    #list of (moving_edge,moving_endpoint,from_edge,to_edge)
    seq_from_1 = []
    seq_from_2 = []
    #TODO keep track of lowest nodes?
    
    
    if start_nodes[3] == '1a':
        up = start_nodes[0]
        moves_network_1,moves_network_2,added_node_network_1,added_node_network_2 = GL_Case1_Tail_with_start(network1,network2,up,isom_2_1,isom_1_2,start_nodes[1:3])
        if not added_node_network_1:
            return False
        
        
    if start_nodes[3] == '1bi':
        up = start_nodes[0]
        moves_network_1,moves_network_2,added_node_network_1,added_node_network_2 = GL_Case1_Tail_with_start(network1,network2,up,isom_2_1,isom_1_2,start_nodes[1:3])
        if not added_node_network_1:
            return False
        
    
    if start_nodes[3] == '1biiA':
        up = start_nodes[0]
        moves_network_1,moves_network_2,added_node_network_1,added_node_network_2 = GL_Case1_Tail_with_start(network1,network2,up,isom_2_1,isom_1_2,start_nodes[1:3])
        if not added_node_network_1:
            return False
    
        
    if start_nodes[3] == '1biiB':
        up = start_nodes[0]
        moves_network_1,moves_network_2,added_node_network_1,added_node_network_2 = GL_Case1_Tail_with_start(network1,network2,up,isom_2_1,isom_1_2,start_nodes[1:3])
        if not added_node_network_1:
            return False
    
        
    if start_nodes[3] == '2a':
        up = start_nodes[0]
        moves_network_2,moves_network_1,added_node_network_2,added_node_network_1 = GL_Case1_Tail_with_start(network2,network1,up,isom_1_2,isom_2_1,start_nodes[1:3])
        if not added_node_network_1:
            return False
        
        
    if start_nodes[3] == '2bi':
        up = start_nodes[0]
        moves_network_2,moves_network_1,added_node_network_2,added_node_network_1 = GL_Case1_Tail_with_start(network2,network1,up,isom_1_2,isom_2_1,start_nodes[1:3])
        if not added_node_network_1:
            return False
    
    if start_nodes[3] == '2biiA':
        up = start_nodes[0]
        moves_network_2,moves_network_1,added_node_network_2,added_node_network_1 = GL_Case1_Tail_with_start(network2,network1,up,isom_1_2,isom_2_1,start_nodes[1:3])
        if not added_node_network_1:
            return False
    
        
    if start_nodes[3] == '2biiB':
        up = start_nodes[0]
        moves_network_2,moves_network_1,added_node_network_2,added_node_network_1 = GL_Case1_Tail_with_start(network2,network1,up,isom_1_2,isom_2_1,start_nodes[1:3])
        if not added_node_network_1:
            return False
    
    
        
#        #use notation as in the paper network1 = N', network2 = N, where ' is denoted p
#        up = lowest_retic_network1
#        moves_network_2,moves_network_1,added_node_network_2,added_node_network_1 = GL_Case1_Tail(network2,network1,up,isom_2_1,isom_1_2)
#        if not added_node_network_1:
#            return False
#    ######################################
#    #Case2: a lowest retic in network2
#    elif lowest_retic_network2:
#        #use notation as in the paper network2 = N', network1 = N, where ' is denoted p
#        up = lowest_retic_network2
#        moves_network_1,moves_network_2,added_node_network_1,added_node_network_2 = GL_Case1_Tail(network1,network2,up,isom_1_2,isom_2_1)
#        if not added_node_network_1:
#            return False        
#    ######################################
#    #Case3: a lowest tree node in network1
#    else:
#        #use notation as in the paper network1 = N, network2 = N'
#        up    = lowest_tree_node_network2
#        moves_network_1,moves_network_2,added_node_network_1,added_node_network_2 = GL_Case3(network1,network2,up,isom_1_2,isom_2_1)
    #Now perform the moves and update the isomorphism
    isom_1_2[added_node_network_1]=added_node_network_2
    isom_2_1[added_node_network_2]=added_node_network_1
    for move in moves_network_1:
        seq_from_1.append((move[0],move[1],From_Edge(network1,move[0],move[1]),move[2]))
        network1 = DoMove(network1,move[0],move[1],move[2],check_validity=False)
    for move in moves_network_2:
        seq_from_2.append((move[0],move[1],From_Edge(network2,move[0],move[1]),move[2]))
        network2 = DoMove(network2,move[0],move[1],move[2],check_validity=False)
    isom_size+=1

    #Do the green line algorithm
    while(isom_size<goal_size):
        #Find lowest nodes above the isom in the networks:
        lowest_tree_node_network1, lowest_retic_network1 = LowestReticAndTreeNodeAbove(network1,isom_1_2.keys(),network2)
        lowest_tree_node_network2, lowest_retic_network2 = LowestReticAndTreeNodeAbove(network2,isom_2_1.keys(),network2)
        
        
        ######################################
        #Case1: a lowest retic in network1
        if lowest_retic_network1:
            #use notation as in the paper network1 = N', network2 = N, where ' is denoted p
            up = lowest_retic_network1
            moves_network_2,moves_network_1,added_node_network_2,added_node_network_1 = GL_Case1_Tail(network2,network1,up,isom_2_1,isom_1_2)
            if not added_node_network_1:
                return False
        ######################################
        #Case2: a lowest retic in network2
        elif lowest_retic_network2:
            #use notation as in the paper network2 = N', network1 = N, where ' is denoted p
            up = lowest_retic_network2
            moves_network_1,moves_network_2,added_node_network_1,added_node_network_2 = GL_Case1_Tail(network1,network2,up,isom_1_2,isom_2_1)
            if not added_node_network_1:
                return False        
        ######################################
        #Case3: a lowest tree node in network1
        else:
            #use notation as in the paper network1 = N, network2 = N'
            up    = lowest_tree_node_network2
            moves_network_1,moves_network_2,added_node_network_1,added_node_network_2 = GL_Case3(network1,network2,up,isom_1_2,isom_2_1)
        #Now perform the moves and update the isomorphism
        isom_1_2[added_node_network_1]=added_node_network_2
        isom_2_1[added_node_network_2]=added_node_network_1
        for move in moves_network_1:
            seq_from_1.append((move[0],move[1],From_Edge(network1,move[0],move[1]),move[2]))
            network1 = DoMove(network1,move[0],move[1],move[2],check_validity=False)
        for move in moves_network_2:
            seq_from_2.append((move[0],move[1],From_Edge(network2,move[0],move[1]),move[2]))
            network2 = DoMove(network2,move[0],move[1],move[2],check_validity=False)
        isom_size+=1
        
    #Add the root to the isomorphism, if it was there
    isom_1_2[root1]=root2
    isom_2_1[root2]=root1	
                    
    #invert seq_from_2, rename to node names of network1, and append to seq_from_1
    return seq_from_1 + ReplaceNodeNamesInMoveSequence(InvertMoveSequence(seq_from_2),isom_2_1)

def GL_Case1_Tail_with_start(N,Np,up,isom_N_Np,isom_Np_N,start_nodes):
    #use notation as in the paper
    #' is denoted p
    xp = Child(Np,up)
    x = isom_Np_N[xp]
    z = start_nodes[0]
    #Case1a: z is a reticulation
    if N.in_degree(z)==2:
        return [],[],z,up
    #Case1b: z is not a reticulation
    #z is a tree node
    if CheckMovable(N,(z,x),z):
        #Case1bi: (z,x) is movable
        #Find a reticulation u in N not in the isomorphism yet
        #TODO: Can first check if the other parent of x suffices here, should heuristcally be better
        u = start_nodes[1]
        v = Child(N,u)
        if v==x:
            return [],[],u,up
        #we may now assume v!=x
        if z==v:
            v = Child(N,z,exclude=[x])
            w = Parent(N,u)
            return [((z,v),z,(w,u))],[],u,up
        w = Parent(N,u,exclude=[z])
        return [((z,x),z,(u,v)),((z,v),z,(w,u))],[],u,up
    #Case1bii: (z,x) is not movable
    c = Parent(N,z)
    d = Child(N,z,exclude=[x])
    #TODO: b does not have to exist if we have an outdeg-2 root, this could be c!
    b = Parent(N,c)
    if N.in_degree(b)!=0:
        #Case1biiA: b is not the root of N
        a = Parent(N,b)
        #First do the move ((c,d),c,(a,b)), then Case1bi applies as (z,x) is now movable
        newN = DoMove(N,(c,d),c,(a,b))
        u = start_nodes[1]
        v = Child(newN,u)
        if v==x:
            #Note: this only happens if u=d, in which case we shouldn't even do the move, but the algorithm says to do it.
            #TODO: catch this by first checking both parents of x for being a reticulation not in the isomorphism yet
            return [((c,d),c,(a,b))],[],u,up
        #we may now assume v!=x
        if z==v:
            #This could happen if z==v and u==b
            v = Child(newN,z,exclude=[x])
            w = Parent(newN,u)
            return [((c,d),c,(a,b)),((z,v),z,(w,u))],[],u,up
        w = Parent(newN,u,exclude=[z])
        return [((c,d),c,(a,b)),((z,x),z,(u,v)),((z,v),z,(w,u))],[],u,up
    #Case1biiB: b is the root of N
    #Note: d is not in the isomorphism
    e = Child(N,d)
    if N.out_degree(x)==2:
        s = Child(N,x)
        t = Child(N,x,exclude=[s])
        if s==e:
            return [((x,t),x,(d,e))],[],d,up
        if t==e:
            return [((x,s),x,(d,e))],[],d,up
        return [((x,s),x,(d,e)),((x,e),x,(z,t)),((x,t),x,(d,s))],[],d,up
    if N.out_degree(e)==2:
        s = Child(N,e)
        t = Child(N,e,exclude=[s])
        if s==x:
            return [((e,t),e,(z,x))],[],d,up
        if t==x:
            return [((e,s),e,(z,x))],[],d,up
        return [((e,s),e,(z,x)),((e,x),e,(d,t)),((e,t),e,(z,s))],[],d,up
    #neither are tree nodes, so both must be leaves
    #In that case, there is no sequence between the two networks.    
    return [],[],False,False


#Find the original location of the moving_endpoint,
#That is, the edge from which we remove it.
def From_Edge(network,moving_edge,moving_endpoint):
    other_parent = Parent(network,moving_endpoint,exclude=moving_edge)
    other_child  =  Child(network,moving_endpoint,exclude=moving_edge)
    return (other_parent,other_child)


def InvertMoveSequence(seq):
    newSeq=[]
    for move in reversed(seq):
        moving_edge,moving_endpoint,from_edge,to_edge = move
        newSeq.append((moving_edge,moving_endpoint,to_edge,from_edge))
    return newSeq

def ReplaceNodeNamesInMoveSequence(seq,isom):
    if type(seq)==int:
        return isom[seq]
    return list(map(lambda x:ReplaceNodeNamesInMoveSequence(x,isom),seq))

def ReplaceNodeNamesByOriginal(network,seq):
    if type(seq)==int:
        return network.node[seq]['original']
    if seq=='rho':
        return "rho"
    return list(map(lambda x:ReplaceNodeNamesByOriginal(network,x),seq))
    

################################################################################
################################################################################
################################################################################
########                                                           #############
########                         Isomorphism                       #############
########                                                           #############
################################################################################
################################################################################
################################################################################



#Checks whether the nodes with the given attributes have the same label
def SameLabels(node1_attributes,node2_attributes):
    return node1_attributes.get('label') == node2_attributes.get('label')
    
#Checks whether two networks are labeled isomorpgic
def Isomorphic(network1,network2):
    return nx.is_isomorphic(network1,network2,node_match=SameLabels)



################################################################################
################################################################################
################################################################################
########                                                           #############
########                         I/O Functions                     #############
########                                                           #############
################################################################################
################################################################################
################################################################################


########
######## Convert Newick to a networkx Digraph with labels (and branch lengths)
########
#Write length newick: convert ":" to "," and then evaluate as list of lists using ast.literal_eval
# Then, in each list, the node is followed by the length of the incoming arc.
# This only works as long as each branch has length and all internal nodes are labeled.
def Newick_To_Network(newick):
    #Ignore the ';'
    newick=newick[:-1]
    #If names are not in string format between ', add these.
    if not "'" in newick and not '"' in newick:
        newick = re.sub(r"\)#H([\d]+)", r",#R\1)", newick)
        newick = re.sub(r"([,\(])([#a-zA-Z\d]+)", r"\1'\2", newick)
        newick = re.sub(r"([#a-zA-Z\d])([,\(\)])", r"\1'\2", newick)
    else:
        newick = re.sub(r"\)#H([d]+)", r"'#R\1'\)", newick)
    newick = newick.replace("(","[")
    newick = newick.replace(")","]")
    nestedtree = ast.literal_eval(newick)
    edges, current_node = NestedList_To_Network(nestedtree,0,1)
    network = nx.DiGraph()
    network.add_edges_from(edges)
    network = NetworkLeafToLabel(network)
    return network
    
#Returns a network in which the leaves have the original name as label, and all nodes have integer names.
def NetworkLeafToLabel(network):
    for node in network.nodes():
        if network.out_degree(node)==0:
            network.node[node]['label']=node
    return nx.convert_node_labels_to_integers(network, first_label=0, label_attribute='original')

#Auxiliary function to convert list of lists to graph 
def NestedList_To_Network(nestedList, top_node, next_node):
    edges = []
    if type(nestedList)==list:
        if type(nestedList[-1])==str and len(nestedList[-1])>2 and nestedList[-1][:2]=='#R':
            retic_node = '#H'+nestedList[-1][2:]
            bottom_node = retic_node
        else:
            bottom_node = next_node
            next_node  += 1
        edges.append((top_node,bottom_node))
        for t in nestedList:
            extra_edges, next_node = NestedList_To_Network(t,bottom_node,next_node)
            edges  = edges + extra_edges
    else:
        if not (len(nestedList)>2 and nestedList[:2]=='#R'):
            edges = [(top_node,nestedList)] 
    return edges, next_node




#Sets the labels of the nodes of a network with a given label dictionary
def Set_Labels(network,label_dict):
    for node, value in label_dict.items():
        network.node[node]['label']=value


def TargetNetwork(network,number_of_moves): #We create a target network by performing a number of random tail_moves and we save the move
    k = number_of_moves
    move_tracker = []
    new_network = copy.deepcopy(network)
    for i in range(0,k):
        #Here we select the fixed number of tail moves that we perform to create the target network.
        selected_move = random.choice(AllValidMoves(new_network,tail_moves=True,head_moves=False))
        move_tracker.append(selected_move)
        new_network = DoMove(new_network,selected_move[0],selected_move[1],selected_move[2],check_validity=True)
    return (new_network,move_tracker)

def DataGeneratornew(Samples,Leaves,Reticulation): #Generates the specified number of sample pairs
    DataSample = []
    for i in range(0,Samples):
        Network = NetworkGenerator(Leaves,Reticulation)
        DataSample.append(Network)
    return DataSample




#Data_samples = []
#for k in range(2,9):
#    for l in range(10,16):
#        network = DataGeneratornew(1,l,k)[0]
#        target = TargetNetwork(network,10)[0]
#        network = NetworkLeafToLabel(network)
#        target  = NetworkLeafToLabel(target)
#        Data_samples.append((network,target))

#network = DataGenerator(1)[0]
#target = TargetNetwork(network,5)[0]
#network = NetworkLeafToLabel(network)
#target  = NetworkLeafToLabel(target)


# This is used to store the data from the 42 test instances in a list of lists.
List_of_samples = []

#k = TMEXPDATA[3]

for k in TMEXPDATA[3:43]:
    #tracker_predict = []
    tracker_predict2 = []
    tracker_random = []
    for i in range(0,10):
         min1 = []
         min2 = []
         min3 = []
         for j in range(0,100):
             try:
        #      testrun1 = Green_Line_Tail_Prediction(network,target)
        #      min1.append(testrun1)
                    testrun2 = Green_Line_Tail_Prediction2(k[0],k[1])
                    min2.append(testrun2)
                    testrun3 = Green_Line_Tail(k[0],k[1])
                    min3.append(testrun3)
             except:
                    continue
        #        
        #        
        #    tracker_predict.append(len(min(min1)))
         tracker_predict2.append(len(min(min2)))
         tracker_random.append(len(min(min3)))
    #     List_of_samples.append((tracker_predict2,tracker_random))
    print(tracker_predict2)
    print(np.asarray(tracker_predict2).mean())
    print(tracker_random)
    print(np.asarray(tracker_random).mean())
    List_of_samples.append((tracker_predict2,np.asarray(tracker_predict2).mean(),tracker_random,np.asarray(tracker_random).mean()))


