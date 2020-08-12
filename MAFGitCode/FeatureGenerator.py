import ete3
import numpy as np
import random as rn
from FPTMAF import *

Data = np.load(r'C:\Users\bryan\Desktop\master thesis\CherryTupledData2.npy', allow_pickle=True)


""" 
We want to create a matrix which holds all the features from the data set and finally also holds the size of
the required search tree to obtain a solution.

"""

def FeatureGenerator(Tuples):
    FeatureMatrix = []
    for i in Tuples:
        root1 = i[0][0][0].get_tree_root()
        root2 = i[0][0][1].get_tree_root()
        Features = []
        Cherry = i[0][1]
        Leaves = len(i[0][0][0].get_leaves()) #Number of leaves
        Depth2 = i[0][0][1].get_distance(root2,i[0][0][1].search_nodes(name=Cherry[0])[0]) #Depth of leaf 1 of cherry in tree2
        Depth3 = i[0][0][1].get_distance(root2,i[0][0][1].search_nodes(name = Cherry[1])[0])#Depth of leaf 2 of cherry in tree2
        Depth  = max(Depth2,Depth3)
        Distance = i[0][0][1].get_distance(i[0][0][1].search_nodes(name=Cherry[0])[0],i[0][0][1].search_nodes(name = Cherry[1])[0])
        Ancestor = i[0][0][1].get_common_ancestor(i[0][0][1].search_nodes(name=Cherry[0])[0],i[0][0][1].search_nodes(name = Cherry[1])[0])
        LCAL = len(Ancestor.get_leaves())
        i = list(i)
        i.append(Leaves)
        i.append(Depth)
        i.append(Distance)
        i.append(LCAL)
        FeatureMatrix.append(i)
    return FeatureMatrix

New_Data2 = FeatureGenerator(Data)
np.save('FeatureMatrix2',New_Data2)            