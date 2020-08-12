# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:01:24 2020

@author: bryan
"""

import ete3
import numpy as np
import random as rn


Data = np.load(r'C:\Users\bryan\Desktop\master thesis\TupledData2.npy', allow_pickle=True)

#We first load in the tree tuples.
#We want to find the cherries and use them to run the MAF
#We tweak the MAF algorithm so that we can manually force the first used cherry

"""
We want to create a function that generates all the features of the tree tuple and the given chosen cherry
We want to find out which cherry is the best pick for the current algorithm by running the plain algorithm
We indicate whether or not we are going to consider the cherry
"""

def get_leaves(tree, leaflist = []):
    for leaf in tree:
        leaflist.append(leaf.name)
    return leaflist



def CherryPicker(T):
    k = 1
    CherrySet = []
    Checked_leaves = []
    leaves = get_leaves(T,[])
    for leaf in leaves:
        if leaf in Checked_leaves:
            continue
        else:
            for j in leaves[k:]:
                if T.get_distance(T.search_nodes(name=leaf)[0],T.search_nodes(name=j)[0]) == 2 and (leaf,j) not in CherrySet:
                    CherrySet.append((leaf,j))
                    Checked_leaves.append(j)
                    k += 1
                    break
                else:
                    continue
            k+=1
    return CherrySet

def CherryTupleGenerator(TreeTuples):
   New_Data_List = []
   for i in TreeTuples:
       CherrySet = CherryPicker(i[0])
       length = len(CherrySet)
       for j in CherrySet:
           New_Data_List.append(((i,j),length))
   return New_Data_List


New_Data2 = CherryTupleGenerator(Data)  

np.save('CherryTupledData2', New_Data2)
