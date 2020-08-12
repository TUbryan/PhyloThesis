import ete3
import numpy as np
import random as rn
import copy


#Here we will implement a Tree Child cherry picking sequence algorithm as seen in the paper.

#Input is a set of phylo trees and a partial cherry picking sequence and an integer k TCS(T,S,k)

#We need a function that checks for forbidden leafs and for trivial cherries.

#First trivial cherries are found by using T/S

Data = np.load(r'C:\Users\bryan\Desktop\master thesis\GeneratedData.npy', allow_pickle=True)
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
        
    
        
def setup(Data):
    global X
    global Cherries
    global Leaves
    X = [i.name for i in Data[1][0].get_leaves()]
    for i in range(len(Data[1])):
        leafset = get_leaves(Data[1][i],[])
        Leaves.append(leafset)
    Cherries = CherryList(Data[1])
    main = TCS(Data[1],[],2)
    return main
        
tracker = 0  
X = []  
Cherries = []   
Leaves = [] 
t = ete3.Tree('((1, (((3, 4), (5, 6)), (((7, 8), 9), ((((2, 10), (11, 12)), (13, 14)), (15, 16))))), ((17, 18), (19, 20)));')
s = ete3.Tree('((((2, (3, 4)), (5, 6)), (((7, 8), 9), (1, (((10, (11, 12)), (13, 14)), (15, 16))))), ((17, 18), (19, 20)));')
Data2 = [3,Data[0][1]]
test = setup(Data2)