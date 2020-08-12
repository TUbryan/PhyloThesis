from __future__ import print_function
import networkx as nx
from RearrDistance_Tools import *
import ast
import os
import sys
import re
import time

##PARAMETERS
filename = None
edges = False
tailMoves = True
headMoves = True
bfSearch = False
simple=False
time_limit=False
##



###############################2. I/O############################
option_help = False
i = 1
while i < len(sys.argv):
    arg= sys.argv[i]
    if arg == "-f" or arg == "--filename":
        i+=1
        filename = str(sys.argv[i])
    if arg == "-tl" or arg == "--timelimit":
        i+=1
        time_limit = float(sys.argv[i])
    if arg == "-e" or arg == "--edges":
        edges = True
    if arg == "-t" or arg == "--tail":
        headMoves = False
    if arg == "-h" or arg == "--head":
        tailMoves = False
    if arg == "-bfs" or arg == "--bfsearch":
        bfSearch = True
    if arg == "-s" or arg == "--simple":
        simple = True
    if arg == "-help" or arg == "--help":
        option_help = True
    i += 1

if len(sys.argv)==1 or option_help:
    print("Mandatory arguments:\n -f or --filename followed by the filename of the file containing two networks \n\nOptional arguments:\n -e or --edges if the input file contains a list of edges in the form [(x1,y1),...,(xn,yn)] with xi and yi integers or strings in the form \"string\". If this option is not selected, the input is assumed to consist of two newick strings.\n -t or --tail for only using tail moves, instead of tail and head moves.\n -h or --head for only using head moves, instead of tail and head moves.\n -bfs or --bfsearch for using a breadth first search to find the an optimal sequence.\n -tl or --timelimit followed by a number of seconds to set a timelimit for the bfs. If no answer is found before the time runs out, the algorithm just returns a lower bound on the distance.\n\nThe output is given as a list of moves in the format:\n  moving_edge, moving_endpoint, to_edge")
    sys.exit()




####################################################
####################################################
####################################################
#############                          #############
#############           MAIN           #############
#############                          #############
####################################################
####################################################
####################################################



test = open(filename, "r")
line1 = test.read()
line1 = line1.split("\n")
test.close()
if edges:
    N = nx.DiGraph()
    M = nx.DiGraph()
    N.add_edges_from(ast.literal_eval(line1[0]))
    M.add_edges_from(ast.literal_eval(line1[1]))
    rootN=Root(N)
    if N.out_degree(rootN)==2:
       N.add_edges_from([('rho',rootN)])
    rootM=Root(M)
    if M.out_degree(rootM)==2:
       M.add_edges_from([('rho',rootM)])
    N = NetworkLeafToLabel(N)
    M = NetworkLeafToLabel(M)
    label_attribute=None
else:
    N = Newick_To_Network(line1[0])
    M = Newick_To_Network(line1[1])
    if not simple:
        print("The networks as list of edges, with node names as used in the computed sequence of moves")
        print("Network 1:")
        for e in N.edges():
            label = N.node[e[1]].get('label')
            if label:
                print(str(e[0])+" "+str(e[1])+" = leaf: "+str(label))
            else:
                print(str(e[0])+" "+str(e[1]))
        print("Network 2:")
        for e in M.edges():
            label = M.node[e[1]].get('label')
            if label:
                print(str(e[0])+" "+str(e[1])+" = leaf: "+str(label))
            else:
                print(str(e[0])+" "+str(e[1]))



if bfSearch:
    if not simple:
        print("Computing a shortest sequence using breadth first search.")
        #Note, the code uses a DFS with incremented depth bound
        #Otherwise, the queue gets too large for memory
    sequence = Depth_First(N,M,tail_moves=tailMoves,head_moves=headMoves,max_time=time_limit,show_bounds=(not simple))
else:
    if not tailMoves:
        if not simple:
            print("Computing a sequence using the `red-line' heuristic.")
        sequence = Red_Line_Head(N,M)
    else: 
        if not simple:
            print("Computing a sequence using the `green-line' heuristic.")
        sequence = Green_Line(N,M,head_moves=headMoves) 

if not simple:
    print("Sequence:")
if sequence==False:
    print("There is no sequence between the networks.")
    sys.exit()
if type(sequence)==int:
    if simple:
        print(";>="+str(sequence)+";?",end='')
    else:
        print("No network was found within the time limit.")
        print("The distance between the networks is at least: "+str(sequence))
    sys.exit()
if edges:
    sequence = ReplaceNodeNamesByOriginal(N,sequence)
    
#Print the output 

   
if simple:
    print(";"+str(len(sequence))+";"+str(sequence),end='')    
else:
    for move in sequence:
        if len(move)==4:
            print(str(move[0])+" "+str(move[1])+" "+str(move[3]))    
        if len(move)==3:
            print(str(move[0])+" "+str(move[1])+" "+str(move[2]))    
    
    

