import networkx as nx
import random
import sys
from copy import deepcopy
from collections import deque
import re
import ast
import time
import numpy as np

# Contents
# - Move functions
# - Finding nodes in a network
# - Sequence finding Functions
# - Isomorphism
# - I/O Functions

TMEXPDATA = np.load(r'C:\Users\bryan\Desktop\master thesis\TMEXPDATA.npy', allow_pickle = True)


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
def LowestReticAndTreeNodeAbove(network,excludedSet):
    lowest_retic = None
    lowest_tree_node = None
    for node in network.nodes():
        if node not in excludedSet:
            for c in network.successors(node):
                if c not in excludedSet:
                    break
            #else runs if the loop was not ended by a break
            #this happens exactly when all of the children are in excludedSet
            else:
                if network.out_degree(node)==2:
                    lowest_tree_node = node
                    if lowest_retic!=None:
                        #break if both types of lowest nodes are found
                        break
                else:
                    lowest_retic = node
                    if lowest_tree_node!=None:
                        #break if both types of lowest nodes are found
                        break
    return lowest_tree_node, lowest_retic


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

def FindRetic(network,excludedSet=[]):
    for node in network.nodes():
        if node not in excludedSet and network.in_degree(node) == 2:
            return node
    return None

def Parent(network,node,exclude=[]):
    for p in network.predecessors(node):
        if p not in exclude:
            return p
    return None
 
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



#Find a sequence by choosing the move that most decreases the upper bound on the number of moves
#This works as long as we can always decrease the bound.
#E.g.1, this upper bound can be the length of the sequence given by Green_Line(N1,N2), the bound can always decrease after one move, if we take the move from the GL sequence (IMPLEMENTED)
#TODO E.g.2, take the upper bound given by this algorithm with bound Green_Line 
def Deep_Dive_Scored(network1,network2,head_moves=True,bound_heuristic=Green_Line):
    if Isomorphic(network1,network2):
        return []
    seq = []
    current_network = network1
    current_best = []
    for move in Green_Line(network1,network2,head_moves=head_moves):
        current_best+=[(move[0],move[1],move[3])]  
    if not current_best:
        return False
    done = False
    current_length=0
    while not done:
        candidate_moves = AllValidMoves(current_network,tail_moves=True,head_moves=head_moves)
        for move in candidate_moves:
            candidate_network = DoMove(current_network,*move)
            if Isomorphic(candidate_network,network2):
                return current_best[:current_length]+[move]
            candidate_sequence = Green_Line(candidate_network,network2,head_moves=head_moves)
            if current_length+len(candidate_sequence)+1<len(current_best):
                current_best = current_best[:current_length]+[move]
                for move2 in candidate_sequence:
                    current_best+=[(move2[0],move2[1],move2[3])]
        next_move = current_best[current_length]
        current_network = DoMove(current_network,*next_move)
        current_length+=1
    return True


def Depth_First(network1,network2,tail_moves=True,head_moves=True,max_time=False,show_bounds=True):
    done = False
    lower_bound = 0
    stop_time = False
    if max_time:
        stop_time = time.time()+max_time
    while not done:
        output = Depth_First_Bounded(network1,network2,tail_moves=tail_moves,head_moves=head_moves,max_depth=lower_bound,stop_time=stop_time)    
        if output=="timeout":
            return lower_bound
        elif type(output)==list:
            return output
        lower_bound+=1
        if show_bounds:
            print(lower_bound)
            
            

#Finds a shortest sequence between network1 and network2 using DFS with bounded depth
def Depth_First_Bounded(network1,network2,tail_moves=True,head_moves=True,max_depth=0,stop_time=False):
    #If we cannot do any moves:
    if not tail_moves and not head_moves:
        if Isomorphic(network1,network2):
            return 0
        else:
            return False
    #Else, make a stack and search
    stack = [[]]
    while stack:
        current_moves = stack.pop()
        current_length = len(current_moves)
        current_network = network1
        for move in current_moves:
            current_network = DoMove(current_network,*move)
        if current_length==max_depth and Isomorphic(current_network,network2):
            return current_moves
        if current_length<max_depth:
            validMoves = AllValidMoves(current_network,tail_moves=tail_moves,head_moves=head_moves)
            for move in validMoves:
                stack.append(current_moves+[move])
        if stop_time and time.time()>stop_time:
            return "timeout"
    return False



#Finds a shortest sequence between network1 and network2 using BFS
def Breadth_First(network1,network2,tail_moves=True,head_moves=True,max_time=False):
    #If we cannot do any moves:
    if not tail_moves and not head_moves:
        if Isomorphic(network1,network2):
            return 0
        else:
            return False
    #Else, make a queue and search
    queue = deque([[]])
    start_time = time.time()
    while queue:
        current_moves = queue.popleft()
        current_network = network1
        for move in current_moves:
            current_network = DoMove(current_network,*move)
        if Isomorphic(current_network,network2):
            return current_moves
        validMoves = AllValidMoves(current_network,tail_moves=tail_moves,head_moves=head_moves)
        for move in validMoves:
            queue.append(current_moves+[move])
        if max_time and time.time()-start_time>max_time:
            return len(current_moves)
    return False



    
def GL_Case1_rSPR(N,Np,up,isom_N_Np,isom_Np_N):
    #use notation as in the paper
    #' is denoted p
    xp = Child(Np,up)
    x = isom_Np_N[xp]
    z = Parent(N,x,exclude=isom_N_Np.keys())
    #Case1a: z is a reticulation
    if N.in_degree(z)==2:
        return [],[],z,up
    #Case1b: z is not a reticulation
    #Find a retic v in N not in the isom yet
    v = FindRetic(N,excludedSet=isom_N_Np.keys())
    u = None
    for parent in N.predecessors(v):
        if CheckValid(N, (parent,v), v, (z,x)):
            u=parent
            return [((u,v),v,(z,x))],[],v,up
    #if none of the moves is valid
    #v should be a reticulation above x
    return [],[],v,up
   
   
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
    
    
def Green_Line_rSPR(network1,network2):
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
        lowest_tree_node_network1, lowest_retic_network1 = LowestReticAndTreeNodeAbove(network1,isom_1_2.keys())
        lowest_tree_node_network2, lowest_retic_network2 = LowestReticAndTreeNodeAbove(network2,isom_2_1.keys())
      
        ######################################
        #Case1: a lowest retic in network1
        if lowest_retic_network1:
            #use notation as in the paper network1 = N', network2 = N, where ' is denoted p
            up = lowest_retic_network1
            moves_network_2,moves_network_1,added_node_network_2,added_node_network_1 = GL_Case1_rSPR(network2,network1,up,isom_2_1,isom_1_2)
        ######################################
        #Case2: a lowest retic in network2
        elif lowest_retic_network2:
            #use notation as in the paper network2 = N', network1 = N, where ' is denoted p
            up = lowest_retic_network2
            moves_network_1,moves_network_2,added_node_network_1,added_node_network_2 = GL_Case1_rSPR(network1,network2,up,isom_1_2,isom_2_1)
        
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
        lowest_tree_node_network1, lowest_retic_network1 = LowestReticAndTreeNodeAbove(network1,isom_1_2.keys())
        lowest_tree_node_network2, lowest_retic_network2 = LowestReticAndTreeNodeAbove(network2,isom_2_1.keys())
        
        
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








#Code used for cases 1 and 2 in the head move red line algorithm
#returns moves1,moves2,addedNode1,addedNode2
def RL_Case1(N1,N2,x_1,isom_N1_N2,isom_N2_N1):
    p_1 = Parent(N1,x_1)
    p_2 = isom_N1_N2[p_1]
    x_2 = Child(N2,p_2,exclude=isom_N2_N1.keys())
    if N2.out_degree(x_2)==2:
        return [],[],x_1,x_2
    elif N2.in_degree(x_2)==2:
        c_2 = FindTreeNode(N2,excludedSet=isom_N2_N1.keys())
        t_2 = Parent(N2,c_2)
        b_2 = Child(N2,c_2)
        #Not in the latex algo, just a minor improvement:
        #If the other child $c_2$ of $p_2$ is a retic, then we can add it.
        if p_2==t_2:
            return [],[],x_1,c_2
        if CheckMovable(N2,(p_2,x_2),x_2):
            q_2 = Parent(N2,x_2,exclude=[p_2]) 
            if x_2==t_2:
                return [],[((q_2,x_2),x_2,(c_2,b_2))],x_1,c_2
            else:
                return [],[((p_2,x_2),x_2,(t_2,c_2)),((t_2,x_2),x_2,(c_2,b_2))],x_1,c_2
        else:
            d_2 = Child(N2,x_2)
            z_2 = Parent(N2,x_2,exclude=[p_2])
            for node in N2.nodes():
                if N2.out_degree(node)==0:
                    parent = Parent(N2,node,exclude=[d_2])
                    if parent:
                        w_2 = parent
                        l_2 = node
                        break
            return [],[((z_2,d_2),d_2,(w_2,l_2)),((z_2,d_2),d_2,(t_2,x_2)),((t_2,x_2),x_2,(c_2,d_2))],x_1,c_2
    else:
        c_2 = FindTreeNode(N2,excludedSet=isom_N2_N1.keys())
        t_2 = Parent(N2,c_2)

        if p_2 == t_2:
            return [],[],x_1,c_2            

        s_2 = None
        r_2 = None
        for node in N2.nodes():
            if N2.in_degree(node)==2:
                for parent in N2.predecessors(node):
                    if parent!=p_2 and CheckMovable(N2,(parent,node),node):
                        s_2 = parent
                        r_2 = node
            if s_2:
                break
        q_2 = Parent(N2,r_2,exclude=[s_2])
        w_2 = Child(N2,r_2)
        
        if r_2 == t_2:
            return [],[((s_2,r_2),r_2,(p_2,x_2)),((p_2,r_2),r_2,(q_2,c_2)),((q_2,r_2),r_2,(s_2,x_2))],x_1,c_2
        if s_2!=t_2:
            return [],[((s_2,r_2),r_2,(p_2,x_2)),((p_2,r_2),r_2,(t_2,c_2)),((t_2,r_2),r_2,(s_2,x_2)),((s_2,r_2),r_2,(q_2,w_2))],x_1,c_2
        else:
            return [],[((s_2,r_2),r_2,(p_2,x_2)),((p_2,r_2),r_2,(s_2,c_2)),((s_2,r_2),r_2,(q_2,w_2))],x_1,c_2       
                    
                        




#Code used for case 3 in the head move red line algorithm
#returns moves1,moves2,addedNode1,addedNode2
def RL_Case3(N1,N2,x_1,isom_N1_N2,isom_N2_N1):
    p_1 = Parent(N1,x_1)
    q_1 = Parent(N1,x_1,exclude=[p_1])
    p_2 = isom_N1_N2[p_1]
    cp_2 = Child(N2,p_2,exclude=isom_N2_N1.keys())
    q_2 = isom_N1_N2[q_1]
    cq_2 = Child(N2,q_2,exclude=isom_N2_N1.keys())
    if cp_2==cq_2:
        return [],[],x_1,cp_2
    elif N2.out_degree(cp_2)==0 and N2.out_degree(cq_2)==0:
        s_2 = None
        r_2 = None
        for node in N2.nodes():
            if N2.in_degree(node)==2 and node not in isom_N2_N1.keys():
                for parent in N2.predecessors(node):
                    if CheckMovable(N2,(parent,node),node):
                        s_2 = parent
                        r_2 = node
            if s_2:
                break
        if s_2==p_2:
            return [],[((s_2,r_2),r_2,(q_2,cq_2)),((q_2,r_2),r_2,(p_2,cp_2))],x_1,r_2     
        else:
            return [],[((s_2,r_2),r_2,(p_2,cp_2)),((p_2,r_2),r_2,(q_2,cq_2))],x_1,r_2     
    else:
        if nx.has_path(N2,cp_2,cq_2) or N2.out_degree(cp_2)==0:
            #Swap p and q
            q_2 , p_2  = p_2  ,q_2
            cp_2, cq_2 = cq_2 ,cp_2
        if CheckMovable(N2,(p_2,cp_2),cp_2):
            return [],[((p_2,cp_2),cp_2,(q_2,cq_2))],x_1,cp_2
        else:
            z = Child(N2,cp_2)
            t = Parent(N2,cp_2,exclude=[p_2])
            return [],[((cp_2,z),z,(q_2,cq_2)),((t,cp_2),cp_2,(z,cq_2))],x_1,z



def Permute_Leaves_Head(network1,network2,isom_1_2,isom_2_1,label_dict_1,label_dict_2):
    print(len(network1.nodes()))
    print(len(isom_1_2))
#    for i,k in isom_1_2.items():
#        print(i,k)
    sequence = []
    #labeldict[label]=leaf
    Y = list(label_dict_1.values())
    cycles = []
    while len(Y)>0:
        y1_1=Y.pop()
        y_2 =isom_1_2[y1_1]
        cycle = [y1_1]
        while network2.node[y_2].get('label')!=network1.node[cycle[0]].get('label'):
            y_new_1 = label_dict_1[network2.node[y_2].get('label')]
#            if len(set(network1.predecessors(cycle[-1]))&set(network1.predecessors(y_new_1)))==0:#cycle[-1] and y_new_1 have NO common parent
#                cycle+=[y_new_1]
            cycle+=[y_new_1]
                
            y_2 = isom_1_2[y_new_1]
            Y.remove(y_new_1)
        if len(cycle)>1:
            cycles+=[list(reversed(cycle))]
    print("cycles",cycles)
    
    t = None
    r = None
    for node in network1:
        if network1.in_degree(node)==2:
           for parent in network1.predecessors(node):
               if CheckMovable(network1,(parent,node),node):
                   t = parent
                   r = node
        if r:
            break
    c_last = Child(network1,r)
    s_last = Parent(network1,r,exclude=[t])


    print("isomorphic", nx.is_isomorphic(network1,network2))

    
    for cycle in cycles:
        c = cycle
        if t in network1.predecessors(cycle[-1]):
            c = [cycle[-1]]+cycle[:-1]
        
        #Skip the first move if the head of the moving arc is already above the last leaf in the cycle

        p_last = Parent(network1,c[-1])
        if r!=p_last:
            move = ((t,r),r,(p_last,c[-1]))
            sequence.append((move[0],move[1],From_Edge(network1,move[0],move[1]),move[2]))
            network1 = DoMove(network1,move[0],move[1],move[2],check_validity=False)
        moved_arc = (t,r)

        c_last_before = c_last

        for i in reversed(range(len(c))):
            p_i   = Parent(network1,r,exclude=[moved_arc[0]])
            p_im1 = Parent(network1,c[i-1])
            if p_i==p_im1:
                print("do nothing, swapping a cherry")
#                move = ((moved_arc[0],r),r,(p_im1,c[i-1]))
#                sequence.append((move[0],move[1],From_Edge(network1,move[0],move[1]),move[2]))
#                network1 = DoMove(network1,move[0],move[1],move[2],check_validity=False)
            else:
                move = ((p_i,r),r,(p_im1,c[i-1]))
                sequence.append((move[0],move[1],From_Edge(network1,move[0],move[1]),move[2]))
                network1 = DoMove(network1,move[0],move[1],move[2],check_validity=False)
                moved_arc = (p_i,r)
            if c[i]==c_last_before:
                c_last = c[i-1]

#        p_0 = Parent(network1,r,exclude=[t])
#        move = ((p_0,r),r,(t,c[-1]))
#        sequence.append((move[0],move[1],From_Edge(network1,move[0],move[1]),move[2]))
#        network1 = DoMove(network1,move[0],move[1],move[2],check_validity=False)

        move = ((t,r),r,(s_last,c_last))
        sequence.append((move[0],move[1],From_Edge(network1,move[0],move[1]),move[2]))
        network1 = DoMove(network1,move[0],move[1],move[2],check_validity=False)
        
        
#    print('last c:', c_last)
    print("isomorphic", nx.is_isomorphic(network1,network2))
    print("labeled isomorphic", Isomorphic(network1,network2))

    return sequence






def Red_Line_Head(network1,network2):
    #Find the root and labels of the networks
    root1 = Root(network1)
    root2 = Root(network2)
    label_dict_1=Labels(network1)
    label_dict_2=Labels(network2)

    #initialize isomorphism
    isom_1_2 = dict()
    isom_1_2[root1]=root2
    isom_2_1 = dict()
    isom_2_1[root2]=root1
    isom_size=1
    
        
    #Check if the roots are of the same type
    if network1.out_degree(root1)!=network2.out_degree(root2):
        return False
        
    #Keep track of the size of the isomorphism and the size it is at the end of the red line algorithm
    goal_size = len(network1)-len(label_dict_1)
    
    #init lists of sequence of moves
    #list of (moving_edge,moving_endpoint,from_edge,to_edge)
    seq_from_1 = []
    seq_from_2 = []
    #TODO keep track of highest nodes?
    

    #Do the red line algorithm
    while(isom_size<goal_size):
        highest_tree_node_network1, highest_retic_network1, highest_leaf_network1 = HighestNodesBelow(network1,isom_1_2.keys())
        highest_tree_node_network2, highest_retic_network2, highest_leaf_network2 = HighestNodesBelow(network2,isom_2_1.keys())
       
        #Case1
        if highest_tree_node_network1:
            moves_network_1,moves_network_2,added_node_network1,added_node_network2 = RL_Case1(network1,network2,highest_tree_node_network1,isom_1_2,isom_2_1)
        #Case2
        elif highest_tree_node_network2:
            moves_network_2,moves_network_1,added_node_network2,added_node_network1 = RL_Case1(network2,network1,highest_tree_node_network2,isom_2_1,isom_1_2)
        #Case3
        else:
            moves_network_1,moves_network_2,added_node_network1,added_node_network2 = RL_Case3(network1,network2,highest_retic_network1,isom_1_2,isom_2_1)
          
          
                
        #Now perform the moves and update the isomorphism
        isom_1_2[added_node_network1]=added_node_network2
        isom_2_1[added_node_network2]=added_node_network1
        for move in moves_network_1:
            seq_from_1.append((move[0],move[1],From_Edge(network1,move[0],move[1]),move[2]))
            network1 = DoMove(network1,move[0],move[1],move[2],check_validity=False)
        for move in moves_network_2:
            seq_from_2.append((move[0],move[1],From_Edge(network2,move[0],move[1]),move[2]))
            network2 = DoMove(network2,move[0],move[1],move[2],check_validity=False)         
        if not nx.is_isomorphic(network1.subgraph(isom_1_2.keys()),network2.subgraph(isom_2_1.keys())):
            print("not unlabeled isom")
            print(seq_from_1)
            print(seq_from_2)
            print(network1.subgraph(isom_1_2.keys()).edges())
            print(network2.subgraph(isom_2_1.keys()).edges())

        isom_size+=1
        
    #TODO Debugging, remove after for speed
    if not nx.is_isomorphic(network1,network2):
        print("not unlabeled isom")
        print(network1.edges())
        print(network2.edges())
    else:
        print("unlabeled isomorphic :)")

    #Add the leaves to the isomorphism
    for node_1 in network1.nodes():
        if network1.out_degree(node_1)==0:
            parent_1 = Parent(network1,node_1)
            parent_2 = isom_1_2[parent_1]
            node_2 = Child(network2,parent_2,exclude=isom_2_1.keys())
            isom_1_2[node_1]=node_2
            isom_2_1[node_2]=node_1

    #Permute the leaves
    seq_permute = Permute_Leaves_Head(network1,network2,isom_1_2,isom_2_1,label_dict_1,label_dict_2)
                    
    #invert seq_from_2, rename to node names of network1, and append to seq_from_1
    return seq_from_1 + seq_permute + ReplaceNodeNamesInMoveSequence(InvertMoveSequence(seq_from_2),isom_2_1)



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



#target = TargetNetwork(network,5)[0]
#network = NetworkLeafToLabel(network)
#target  = NetworkLeafToLabel(target)
#tracker_predict = []

Sample_tracker = []
trackertje = 0

for i in TMEXPDATA:
    testrun1 = Green_Line_Tail(i[0],i[1])
    Sample_tracker.append(len(testrun1))
    trackertje+=1
    print(trackertje)
    
    
    
#print(len(testrun1))

