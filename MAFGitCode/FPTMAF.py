# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:16:40 2020

@author: bryan
"""

import ete3
import numpy as np 
import random as rn

"""
We first need to determine the cherries in both trees, because we need those
for the algorithm. We determine the distances from leaf to leaf

"""

def get_leaves(tree, leaflist = []):
    for leaf in tree:
        leaflist.append(leaf.name)
    return leaflist


names = list(np.arange(0,10000,1))
Data = np.load(r'C:\Users\bryan\Desktop\master thesis\TupledTestData.npy', allow_pickle = True)

for i in range(0,len(names)):
    names[i] = str(names[i])

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
        
def CherryFinder(T, leaves = []):
    Found = False
    if leaves == []:
        leaves = get_leaves(T,[])
        leaf = rn.choice(leaves)
        leaves.remove(leaf)
        for i in leaves:
            if T.get_distance(T.search_nodes(name=leaf)[0],T.search_nodes(name=i)[0]) == 2:
                Found = True
                return (leaf,i)
            else:
                continue
        if Found == False:
            return CherryFinder(T,leaves)
    else:
        leaf = rn.choice(leaves)
        leaves.remove(leaf)
        for i in leaves:
            if T.get_distance(T.search_nodes(name=leaf)[0],T.search_nodes(name=i)[0]) == 2:
                Found = True
                return (leaf,i)
            else:
                continue
        if Found == False:
            return CherryFinder(T,leaves)
            
          
"""n is the parameter that determines the population size"""        
def TreeMaker(n): 
    t = ete3.Tree()
    t.populate(n, names_library = names)
    i = 0
    for n in t.traverse():
        if n.name == '':
            n.name = str(i)
        i +=1
    return t




""" 
Input is two phylogenetic trees T1 and T2 and an integer k
We want to know if there exists an agreement forest of at most k+1 components, i.e, 
there exist k cuts such that we create an agreement forest,
"""


""" Recursively branch on new trees"""


"""We can have multiple trees (forest) the main tree is always the first tree in the list"""



def MAF(T,U,k):
    global counter
    F = [i.copy() for i in U]
    J = [i.copy() for i in T]
    if (max(F,key=len).compare(J[0]).get('rf') == 0.0 and k >= 0):
           return F
    if all(len(i) ==1 for i in F):
        return F
    if k < 0:
        return False
#    """Figure out if we have a singleton in F that we can delete from T
    for i in F:
        if len(i) == 1:
            candidate = J[0].search_nodes(name = i.name)
            if candidate == []:
                continue
            else:
                candidate[0].delete()
    # Find a random cherry in T
    cherry = CherryFinder(J[0],leaves = [])
    #Determine if the cherry also occurs in F, if so collapse cherry into a single leaf in T and F
    for i in range(0,len(F)):
        try:
            if F[i].get_distance(cherry[0],cherry[1]) ==2:
               J[0].search_nodes(name = cherry[1])[0].name = cherry[0]+cherry[1]
               pointer1 = J[0]&cherry[0]
               pointer1.delete()
               F[i].search_nodes(name = cherry[1])[0].name = cherry[0]+cherry[1]
               p1 = F[i]&cherry[0]
               p1.delete()
               return MAF(J,F,k)
        except:
            continue
    #Else we need to consider 3 cases
    G = [i.copy() for i in F]
    H = [i.copy() for i in F]
    I = [i.copy() for i in F]
        
    #Case 1
    J1 = [i.copy() for i in J]
    P = False
    for i in range(0,len(G)):
        try:
            first = G[i].search_nodes(name= cherry[0])[0]
            first.detach()
            G.append(first)
            counter +=1
            P = MAF(J1,G,k-1)
            break
        except:
            continue
        
    #Case 2
    J2 = [i.copy() for i in J]
    Q = False
    for i in range(0,len(H)):
        try:
            second = H[i].search_nodes(name= cherry[1])[0]
            second.detach()
            H.append(second)
            counter +=1
            Q = MAF(J2,H,k-1)
            break
        except:
            continue
        
    #Case 3
    J3 = [ i.copy() for i in J]
    R = False
    #First we check if both nodes are in the same tree and if so we cut edges leaving the path.
    par = 0
    for i in range(0,len(I)):
        try :
            cherry1 = I[i].search_nodes(name=cherry[0])[0]
            cherry2 = I[i].search_nodes(name= cherry[1])[0]
            common = I[i].search_nodes(name= I.get_common_ancestor(cherry1,cherry2))[0].name
            while cherry1.up.name != common:
                sister = I[i].search_nodes(name = cherry1.name)[0].get_sisters()[0]
                sister.detach()
                I.append(sister)
                cherry1 = cherry1.up
                par +=1
            while cherry2.up.name != common:
                sister = I[i].search_nodes(name = cherry2.name)[0].get_sisters()[0]
                sister.detach()
                I.append(sister)
                par +=1
            counter += 1
            R = MAF(J3,I,k-par)
        except:
             continue  
    
    if P != False:
        return P
    if Q != False: 
        return Q
    if R != False:
        return R
    if P == False and Q == False and R == False:
        return 'No agreementforest'
    








counter = 0
t = TreeMaker(6)
l = TreeMaker(6)
#print(t)
#print(l)
#solution = MAF([t],[l],2)
#print(solution)
solution = MAF([Data[0][0]],[Data[0][1]],15)

