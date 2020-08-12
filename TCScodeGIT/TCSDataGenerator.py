import ete3
import numpy as np
import random as rn
import networkx as nx
import copy


'''
The input will be trees and the output will be features. Later we will determine 
which samples are considered good and which are considered bad.
'''

Data = np.load(r'C:\Users\bryan\Desktop\master thesis\GeneratedData.npy',allow_pickle=True)



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




List_of_Features = [] #List of features for all the cherries regarding one set of phylo trees.


def FeatureGenerator(T,S=[]):
    List_of_Features = []
    Leafset = []
    X = [i.name for i in T[0].get_leaves()]
    for i in range(len(T)):
        leafset = get_leaves(T[i],[])
        Leafset.append(leafset)
    F = copy.deepcopy(T)
    H = copy.deepcopy(S)
    Cherries = CherryList(F)
    #We first make sure to find the trivial cherries before we generate features
    #This is because we will never have to consider trivial cherries for choice.    
    while TrivialCherryChecker(F,H,Cherries) != False:
        Cherry = TrivialCherryChecker(F,H,Cherries)
        H.append(Cherry)
        store = updater(F,H,Cherries,Leafset)
        F = store[0]
        Cherries = store[1]
        Leafset = store[2]
    New_cherries = [[] for n in Cherries]
    for i in range(len(Cherries)):
        for j in Cherries[i]:
           New_cherries[i].append(j)
           New_cherries[i].append(j[::-1])
           
           
    List_of_Cherries = list(set().union(*New_cherries))
    List_of_Features = [[] for i in range(len(List_of_Cherries))]
    
    
    for i in range(len(List_of_Cherries)):
        tracker = 0
        for j in range(len(New_cherries)):
            if List_of_Cherries[i] in New_cherries[j]:
                tracker +=1
        List_of_Features[i].append(List_of_Cherries[i])
        List_of_Features[i].append(tracker/len(F)) #in how many trees does the cherry appear, averaged by #trees
        
        
    for i in range(len(List_of_Cherries)):
        tracker = 0
        for j in range(len(New_cherries)):
            for k in New_cherries[j]:
                if k[0] == List_of_Cherries[i][0] and k[1] != List_of_Cherries[i][1]:
                    tracker +=1
        List_of_Features[i].append(tracker/len(F)) #In how many trees does x from (x,y) appear in different cherry. #trees average
        
        
    for i in range(len(List_of_Cherries)):
        tracker = 0
        for j in range(len(F)):
            if List_of_Cherries[i] in New_cherries[j]:
                tracker +=2
            else:
                pointer1 = F[j].search_nodes(name = List_of_Cherries[i][0])[0]
                pointer2 = F[j].search_nodes(name = List_of_Cherries[i][1])[0]
                distance = F[j].get_distance(pointer1,pointer2)
                tracker += distance
        List_of_Features[i].append(tracker/len(F)) #Average distance among all the trees
        
        
    for i in range(len(List_of_Cherries)):
        tracker = 0
        for j in range(len(F)):
            if List_of_Cherries[i] in New_cherries[j]:
                tracker +=2
            else:
                pointer1 = F[j].search_nodes(name = List_of_Cherries[i][0])[0]
                pointer2 = F[j].search_nodes(name = List_of_Cherries[i][1])[0]
                pointer3 = F[j].get_common_ancestor(pointer1,pointer2)
                tracker += len(pointer3)
        List_of_Features[i].append(tracker/len(F)) #Average size of subtree of LCA among all trees.
        
        
    return List_of_Features,H
    
    
def setup(Data):
    FeatureData = []
    for i in range(len(Data)):
        Total = FeatureGenerator(Data[i][1],[])
        Features = Total[0]
        S = Total[1]
        FeatureData.append((Data[i],S,Features))
    return FeatureData
    
test = setup(Data)        
    