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

Data = np.load(r'C:\Users\bryan\Desktop\master thesis\TCSProgram\ReticulationTestData\7ret.npy', allow_pickle=True)
Dec_Tree = pickle.load(open('TCSDecTree.sav','rb'))

names = list(np.arange(0,10000))


def TreeMaker(n): 
    t = ete3.Tree()
    t.populate(n, names_library = names)
    i = 0
    for n in t.traverse():
        if n.name == '':
            n.name = str(i)
        i +=1
    return t

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



def CherryList(T):
    CherryList = []
    for i in range(0,len(T)):
        cherries = CherryPicker(T[i])
        CherryList.append(cherries)
    return CherryList
        
#Here T is the set of phylogenetic trees.

def TrivialCherryChecker(T,S,Cherries):
    S = S
    Cherries = Cherries
    Trivial_Cherry = False
    for i in Cherries[0]:
        forbidden = ForbiddenCherryChecker(i[1],S)
        if forbidden == True:
            forbidden = ForbiddenCherryChecker(i[0],S)
            if forbidden == True:
                continue
            tracker = all(i in j for j in Cherries)
            if tracker == True:
                Trivial_Cherry = i[::-1]        
        tracker = all(i in j for j in Cherries)
        if tracker == True:
            Trivial_Cherry = i
            return Trivial_Cherry
        else:
            continue
    return Trivial_Cherry
            
def ForbiddenCherryChecker(y,sequence):
    if any(y == i[0] for i in sequence):
        return True
    else:
        return False
    
def updater(T,S,List_of_cherries,Leaves):
    Leaves = Leaves
    for f in range(len(S)):
        Target = S[f]
        for i in range(0,len(T)):
            if (Target in List_of_cherries[i]) or (Target[::-1] in List_of_cherries[i]):
                pointer1 = T[i].search_nodes(name = Target[0])[0]
                pointer2 = T[i].search_nodes(name = Target[1])[0]
                pointer1.detach()
                Leaves[i].remove(pointer1.name)
                for j in T[i].traverse():
                   if len(j.get_children()) == 1:
                       j.delete()
                if pointer2.get_sisters() == []:
                    New = List_of_cherries[i]
                    try:
                       New.remove(Target)
                    except:
                        New.remove(Target[::-1])
                    List_of_cherries[i] = New
                if pointer2.get_sisters() != []:
                   if pointer2.get_sisters()[0].name in Leaves[i]:
                       replacer = (pointer2.name,pointer2.get_sisters()[0].name)
                       List_of_cherries[i]=[replacer if (x==Target or x==Target[::-1]) else x for x in List_of_cherries[i]]
                   else:
                       New = List_of_cherries[i]
                       try:
                          New.remove(Target)
                       except:
                          New.remove(Target[::-1])
                       List_of_cherries[i] = New
    return T,List_of_cherries,Leaves

def TCS(T,S,k):
    global tracker
    global X
    global Cherries
    global Leaves
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
#        print('hier')
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
        for i in C:
            Z = copy.deepcopy(H)
            if ForbiddenCherryChecker(i[1],Z) == False:
                Z.append(i)
                Stemp = TCS(T,Z,k)
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
        return Sopt    
    
    
def TimeTCS(T,S,k,t):
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
        for i in C:
            try:
                if time.time()-t >1800:
                    Sopt = None
                    break
            except:
                pass
            Z = copy.deepcopy(H)
            if ForbiddenCherryChecker(i[1],Z) == False:
                Z.append(i)
                Stemp = TimeTCS(T,Z,k,t)
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
        return Sopt
 

"""
Remember to return t into the alg if we want to do a timelimit
"""    
        
def DecisionTCS(T,S,k,timer):
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
        for i in C:
            try:
               if time.time()-t >300:
                  Sopt == None                 
                  break
            except:
                pass
            predictor = False
            if prediction(i,T) == 1:
                predictor = True
                Z = copy.deepcopy(H)
                if ForbiddenCherryChecker(i[1],Z) == False:
                    Z.append(i)
                    Stemp = DecisionTCS(T,Z,k,t)
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
        predictor = False
        if predictor == False:
            for i in C:
                try:
                   if time.time()-t >600:
                       Sopt == None
                       break
                except:
                   pass
                Z = copy.deepcopy(H)
                if ForbiddenCherryChecker(i[1],Z) == False:
                    Z.append(i)
                    Stemp = DecisionTCS(T,Z,k,t)
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
        return Sopt    
 

def prediction(i,Trees):
    Features = FeatureGetter(Trees,i)
    pred = Dec_Tree.predict((np.array(Features).reshape(1,-1)))
    return pred
    
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
        predictor = prediction2(C,T)
        C_sorted = [x for _, x in sorted(zip(predictor,C), key=lambda pair: pair[0])][::-1]
        for i in C_sorted:
            try:
               if time.time()-t >1800:
                  Sopt = None                 
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
 

def prediction2(C,Trees):
    All_Features = []
    for i in C:
       Features = FeatureGetter(Trees,i)
       All_Features.append(Features)
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


"""
To use regular TCS alg, change main to TCS instead of DecisionTCS
"""    
       
def setup(Data):
    Datas = [[] for n in range(0,10)]
    global X
    global Cherries
    global Leaves
    global timer
    global tracker
    global monitor
    for i in range(0,10):
       tracker = 0
       X = []
       Leaves = []
       X = [i.name for i in Data[i][1][0].get_leaves()]
       Cherries = CherryList(Data[i][1])
       for j in range(len(Data[i][1])):
           leafset = get_leaves(Data[i][1][j],[])
           Leaves.append(leafset)
       timer = time.time()
       main = TimeTCS(Data[i][1],[],Data[i][0],timer)
       timer = time.time()-timer
       Datas[i].append((main,tracker,timer))
       monitor +=1
       print(monitor)
    return Datas
 
def setup2(Data):
    Datas = [[] for n in Data]
    global X
    global Cherries
    global Leaves
    global tracker
    global monitor
    for i in range(0,25):
       tracker = 0
       X = []
       Leaves = []
       X = [i.name for i in Data[i][1][0].get_leaves()]
       Cherries = CherryList(Data[i][1])
       for j in range(len(Data[i][1])):
           leafset = get_leaves(Data[i][1][j],[])
           Leaves.append(leafset)
       timer = time.time()
       main = DecisionTCS(Data[i][1],[],Data[i][0],timer)
       timer = time.time()-timer
       Datas[i].append((main,tracker,timer))
       monitor +=1
       print(monitor)
    return Datas

def setup3(Data):
    Datas = [[] for n in range(0,10)]
    global X
    global Cherries
    global Leaves
    global tracker
    global monitor
    for i in range(0,10):
       tracker = 0
       X = []
       Leaves = []
       X = [i.name for i in Data[i][1][0].get_leaves()]
       Cherries = CherryList(Data[i][1])
       for j in range(len(Data[i][1])):
           leafset = get_leaves(Data[i][1][j],[])
           Leaves.append(leafset)
       timer = time.time()
       main = NewDecisionTCS(Data[i][1],[],Data[i][0],timer)
       timer = time.time()-timer
       Datas[i].append((main,tracker,timer))
       monitor +=1
       print(monitor)
    return Datas





       
tracker = 0
monitor = 0  
timer = 0
X = []  
Cherries = []   
Leaves = [] 

def RunFunction(Data):
    Data1 = setup(Data)
#    Data2 = setup2(Data)
    Data3 = setup3(Data)
    New_Data = []
    New_Data.append(Data1)
#    New_Data.append(Data2)
    New_Data.append(Data3)
    return New_Data



testrun = RunFunction(Data)
np.save('7retLongResults',testrun)
testje1 = []
testje2 = []
for i in range(0,10):
    testje1.append(testrun[0][i][0][2])
    testje2.append(testrun[1][i][0][2])



