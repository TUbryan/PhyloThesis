import ete3
import numpy as np
import random as rn
from FPTMAF import *
from rSPRsimulator import *


"""
We want to generate many trees/forests that we can use to hopefully train our decision tree

We first run the program and keep saving the first tree/forest duo we encounter and run the MAF again, also we run the program with
every cherry within the start tree to generate as much data as possible.

"""

#We start by generating 1000 trees all sporting 10 leaves
#We make sure we have 1000 different trees


names = list(np.arange(0,10000,1))
names = [str(i) for i in names]
#starttree = TreeMaker(10)
#Tree_List = [starttree]

def TreeListGenerator(n):
#    trace = 0
#    starttree = TreeMaker(rn.randint(20,30))
#    Tree_List = [starttree]
#    for i in range(0,n-1):
#        t = TreeRandomizer(starttree,rn.randint(2,4))[1]
#        Tree_List.append(t)
#        trace +=1
#        print(trace)
    trace = 0
    Tree_List = []
    for i in range(0,n-1):
        t = TreeMaker(rn.randint(20,25))
        Tree_List.append(t)
        trace +=1
        print(trace)
    return Tree_List
        
    
    

def check(Tree, Tree_List):
    check = []
    for i in Tree_List:
        if Tree.compare(i).get('rf') != 0:
            check.append(False)
        else:
            check.append(True)
    if True in check:
        return True
    else:
        return False


#After we have created the different trees we will create a counterpart of each tree by performing 5 rSPR moves.
#We then are left with a list of tuples containing our start trees for the MAF algorithm.

def TreeTupleListGenerator(Tree_list):
    tracker = 0
    Tree_Tuple_List = []
    for i in range(0,len(Tree_list)):
        Tree_Tuple = TreeRandomizer(Tree_list[i],rn.randint(5,12))
        Tree_Tuple_List.append(Tree_Tuple)
        tracker += 1
        print(tracker)
    return Tree_Tuple_List
    
    
    
    
   
Data = TreeListGenerator(30)
TupledTestData = np.array(TreeTupleListGenerator(Data))   
  

np.save('LargeTupledTestData', TupledTestData)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    