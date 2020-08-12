import numpy as np
from RearrDistance_Tools import *
from TMNetworkGenerator import *
import copy
from operator import itemgetter

test = DataGenerator(1)


#run1 = Breadth_First(test[0][0],test[0][1],tail_moves=True,head_moves=False,max_time=300)
#run = Green_Line(test[0][0],test[0][1],head_moves=False)


#AllValidMoves(graph,tail_moves=True,head_moves=False)
#DoMove(graph, moving_edge,moving_endpoint,to_edge,check_validity=True)


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
 

       

def setup(network,target_network):
    Dataset = []
#    target_network, moves = TargetNetwork(network,3)
    list_of_moves = AllValidMoves(network,tail_moves=True,head_moves=False)
#    list_of_moves.remove(moves[0])
    target_network = NetworkLeafToLabel(target_network)
    for i in list_of_moves:
        new_network = NetworkLeafToLabel(network)
        new_network = DoMove(network,i[0],i[1],i[2],check_validity=True)
#        sequencefinder = Breadth_First(new_network,target_network,tail_moves=True,head_moves=False,max_time=100)
        sequencefinder = Green_Line(new_network,target_network,head_moves=False)
        if type(sequencefinder) == int:
            Dataset.append((i,sequencefinder,sequencefinder))
        else:
            Dataset.append((i,sequencefinder,len(sequencefinder)))
#        print(tracker)
    Dataset = sorted(Dataset, key=itemgetter(2))
    return Dataset


#Network = DataGenerator(1)
#target,moves = TargetNetwork(Network[0],3)
#testrun = setup(Network[0],target)


        