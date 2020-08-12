import ete3
import numpy as np
import random as rn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
import copy
import time
import pickle


#Here we will implement a Tree Child cherry picking sequence algorithm as seen in the paper.

#Input is a set of phylo trees and a partial cherry picking sequence and an integer k TCS(T,S,k)

#We need a function that checks for forbidden leafs and for trivial cherries.

#First trivial cherries are found by using T/S

Data = np.load(r'C:\Users\bryan\Desktop\master thesis\TCSTestData.npy', allow_pickle=True)
Dec_Tree = pickle.load(open('TCSDecTree.sav','rb'))


def NewDecisionTCS(T,S,k,timer):
    global tracker
    global X
    global Cherries
    global Leaves
    t = timer
    tracker +=1
    Leafset = copy.deepcopy(Leaves)
    Cherryset = copy.deepcopy(Cherries)
    F = copy.deepcopy(T)
    H = copy.deepcopy(S)
    store = updater(F,H,Cherryset,Leafset)
    F = store[0]
    Cherryset = store[1]
    Leafset = store[2]
    while TrivialCherryChecker(F,H,Cherryset) != False:
        Cherry = TrivialCherryChecker(F,H,Cherryset)
        H.append(Cherry)
        store = updater(F,H,Cherryset,Leafset)
        F = store[0]
        Cherryset = store[1]
        Leafset = store[2]
    #store = updater(F,H,Cherryset,Leafset)
    #F = store[0]
    #Cherryset = store[1] 
    #Leafset = store[2]
    for i in range(len(F)):
        for j in range(len(Cherryset[i])):
            if ForbiddenCherryChecker(Cherryset[i][j][1],S) == True and ForbiddenCherryChecker(Cherryset[i][j][0],S) == True:
                return None
    n = len(set().union(*Leafset))
    l = len(H)- len(X) + n
    C = list(set().union(*Cherryset))
#    if C == [] and len(S)-len(X)+l < k:
#        return False
    for i in range(len(C)):
        C.append(C[i][::-1])
    if len(C) == 0:
        x = list(set().union(*Leafset))[0]
        H.append((x,''))
        return H
    elif (len(C) > 8*k) or (l > k):
        return None
    else:
        Sopt = None
        
        predictor = prediction(C,T)
        C_sorted = [x for _, x in sorted(zip(predictor,C), key=lambda pair: pair[0])]
        for i in C_sorted:
            try:
               if time.time()-t >600:
                  Sopt == None                 
                  break
            except:
                pass
            Z = copy.deepcopy(H)
            if ForbiddenCherryChecker(i[1],Z) == False:
                Z.append(i)
                Stemp = NewDecisionTCS(T,Z,k,t)
                if Stemp == None:
                    WeightStemp = 1000000000
                else:
                    WeightStemp = len(Stemp) - len(X)
                if Sopt == None:
                    Weightopt = 1000000000
                else:
                    Weightopt = len(Sopt)-len(X)
                if WeightStemp < Weightopt:
                    Sopt = Stemp
                try:
                    if Weightopt <= k: 
                        break
                except:
                    pass
            else:
                continue
#        predictor = False
#        if predictor == False:
#            for i in C:
#                try:
#                   if time.time()-t >600:
#                       Sopt == None
#                       break
#                except:
#                   pass
#                Z = copy.deepcopy(H)
#                if ForbiddenCherryChecker(i[1],Z) == False:
#                    Z.append(i)
#                    Stemp = DecisionTCS(T,Z,k,t)
#                    if Stemp == None:
#                        WeightStemp = 1000000000
#                    else:
#                        WeightStemp = len(Stemp) - len(X)
#                    if Sopt == None:
#                        Weightopt = 1000000000
#                    else:
#                        Weightopt = len(Sopt)-len(X)
#                    if WeightStemp < Weightopt:
#                        Sopt = Stemp
#                    try:
#                       if Weightopt <= k:
#                           break
#                    except:
#                        pass
        return Sopt    
 

def prediction(C,Trees):
    All_Features = []
    for i in C:
       Features = FeatureGetter(Trees,i)
       All_Features.append()
    pred = Dec_Tree.predict((np.array(All_Features)))
    return pred
    


def FeatureGetter(Trees, cherry):
    features = []
    Cherry = cherry
    
    tracker = 0
    for i in Trees:
        if i.get_distance(Cherry[0],Cherry[1]) == 2:
            tracker +=1
    features.append(tracker/len(Trees))
    
    tracker = 0
    for i in Trees:
        for k in i.get_leaves():
            if i.get_distance(Cherry[0],k) == 2 and k != Cherry[1]:
                tracker += 1
    features.append(tracker/len(Trees))
    
    distance = 0
    for i in Trees:
        distance += i.get_distance(Cherry[0],Cherry[1])
    features.append(distance/len(Trees))
    
    size = 0
    for i in Trees:
        pointer1 = i.search_nodes(name = Cherry[0])[0]
        pointer2 = i.search_nodes(name = Cherry[1])[0]
        if i.get_distance(pointer1,pointer2) == 2:
            size +=2
        else:
            pointer3 = i.get_common_ancestor(pointer1,pointer2)
            size += len(pointer3)
    features.append(size/len(Trees))
    return features