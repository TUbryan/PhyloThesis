import time
import ete3
import numpy as np
import random as rn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
import pickle
import matplotlib.pyplot as plt

Data = np.load(r'C:\Users\bryan\Desktop\master thesis\LargeTupledTestData.npy',allow_pickle = True)
Dec_Tree = pickle.load(open('DecisionTree3.sav','rb'))

names = list(np.arange(0,10000,1))
for i in range(0,len(names)):
    names[i] = str(names[i])
    
   
    
#We want to test our algorithms on bigger samples, but bigegr samples take a lot more time therefor we want to build a breaker into it.
        
    
       
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

def BaseMAF(T,U,k,t):
    t = t
    while time.time()-t < 150:
        global counter
        F = [i.copy() for i in U]
        J = [i.copy() for i in T]
        for i in range(0,len(F)):
            if len(F[i])>1 and (J[0].compare(F[i]).get('rf') == 0.0 and k >= 0):
                return time.time() - t
        if all(len(i) ==1 for i in F):
            return time.time()-t
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
                   return BaseMAF(J,F,k,t)
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
                P = BaseMAF(J1,G,k-1,t)
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
                Q = BaseMAF(J2,H,k-1,t)
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
                R = BaseMAF(J3,I,k-par,t)
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
    return 150
    
    
def DecisionMAF(T,U,k,t):
    t = t
    while time.time() - t < 150:
        global counter2
        F = [i.copy() for i in U]
        J = [i.copy() for i in T]
        for i in range(0,len(F)):
            if (len(F[i])>1 and J[0].compare(F[i]).get('rf') == 0.0 and k >= 0):
                return time.time() - time
        if all(len(i) ==1 for i in F):
            return time.time() - t
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
        FeatureMatrix = FeatureGetter(J,F)
        cherry = Predictor(FeatureMatrix)
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
                   return DecisionMAF(J,F,k,t)
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
                counter2 +=1
                P = DecisionMAF(J1,G,k-1,t)
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
                counter2 +=1
                Q = DecisionMAF(J2,H,k-1,t)
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
                counter2 += 1
                R = DecisionMAF(J3,I,k-par,t)
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
    return 150
    
#The following function finds the features required for     
    
def FeatureGetter(T,F):
    Features=[]
    Cherries = CherryPicker(T[0])
    leaves = len(T[0].get_leaves())
    NumCherries = len(Cherries)
    root = T[0].get_tree_root()
    Distance = 10e10 #Simulate infinty
    LCAL = 10e10 #Simulate infinity
    for i in Cherries:
        CherryFeature = []
        Depth = T[0].get_distance(root,T[0].search_nodes(name=i[0])[0])
        for j in F:
            try:
                pointer1 = j.search_nodes(name=i[0])[0]
                pointer2 = j.search_nodes(name=i[1])[0]
                Distance = j.get_distance(pointer1,pointer2)
                LCA = j.get_common_ancestor(pointer1,pointer2)
                LCAL = len(LCA.get_leaves)
            except:
                continue
        CherryFeature.append(i)
        CherryFeature.append(NumCherries)
        CherryFeature.append(leaves)
        CherryFeature.append(Depth)
        CherryFeature.append(Distance)
        CherryFeature.append(LCAL)
        Features.append(CherryFeature)
    return Features
            
       
def Predictor(X):
    for i in X:
        if Dec_Tree.predict(np.asarray(i[1:]).reshape(1,-1)) == 1:
            return i[0]
        else:
            continue
    return rn.choice(X)[0]

    
def CherryMAF(T,U,k,t):
    t = t
    while time.time() - t < 150:
        global counter3
        F = [i.copy() for i in U]
        J = [i.copy() for i in T]
        for i in range(0,len(F)):
            if (len(F[i])>1 and J[0].compare(F[i]).get('rf') == 0.0 and k >= 0):
                return time.time() - t
        if all(len(i) ==1 for i in F):
            return time.time()-t
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
        cherry = SmallestCherry(J[0])
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
                   return CherryMAF(J,F,k,t)
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
                counter3 +=1
                P = CherryMAF(J1,G,k-1,t)
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
                counter3 +=1
                Q = CherryMAF(J2,H,k-1,t)
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
                counter3 += 1
                R = CherryMAF(J3,I,k-par,t)
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
    return 150

def SmallestCherry(T):
    root = T.get_tree_root()
    cherries = CherryPicker(T)
    depthlist = []
    for i in range(0,len(cherries)):
        depth = T.get_distance(T.search_nodes(name = cherries[i][0])[0],root)
        depthlist.append(depth)
    index = np.argmin(depthlist)
    return cherries[index]



def LimitTimeFunction(Data):
    TimeMatrix = [[],[],[]]
    global counter
    global counter2
    global counter3
    for i in range(0,len(Data)):
       counter = 0
       counter2 = 0
       counter3 = 0
       start = time.time()
       first = BaseMAF([Data[i][0]],[Data[i][1]],30,start)
       TimeMatrix[0].append((counter,first))
       start = time.time()
       second = DecisionMAF([Data[i][0]],[Data[i][1]],30,start)
       TimeMatrix[1].append((counter2,second))
       start = time.time()
       third = CherryMAF([Data[i][0]],[Data[i][1]],30,start)
       TimeMatrix[2].append((counter3,third))
    return TimeMatrix

       

Attempt = LimitTimeFunction(Data)

