import ete3
import numpy as np
import random as rn

Data = np.load(r'C:\Users\bryan\Desktop\master thesis\FeatureMatrix2.npy', allow_pickle = True) 




def get_leaves(tree, leaflist = []):
    for leaf in tree:
        leaflist.append(leaf.name)
    return leaflist


names = list(np.arange(0,10000,1))

for i in range(0,len(names)):
    names[i] = str(names[i])
        
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







def MAF(T,U,k,cherry):
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
    if cherry == []:
       cherry = CherryFinder(J[0],leaves = [])
    else:
        cherry = cherry
    #Determine if the cherry also occurs in F, if so collapse cherry into a single leaf in T and F
    for i in range(0,len(F)):
        try:
            if F[i].get_distance(cherry[0],cherry[1]) == 2:
               J[0].search_nodes(name = cherry[1])[0].name = cherry[0]+cherry[1]
               pointer1 = J[0]&cherry[0]
               pointer1.delete()
               F[i].search_nodes(name = cherry[1])[0].name = cherry[0]+cherry[1]
               p1 = F[i]&cherry[0]
               p1.delete()
               return MAF(J,F,k,[])
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
            counter+=1
            P = MAF(J1,G,k-1,[])
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
            Q = MAF(J2,H,k-1,[])
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
            counter +=1
            R = MAF(J3,I,k-par,[])
        except:
            continue  
    
    if P != False:
        return P
    if Q != False: 
        return Q
    if R != False:
        return R
    if P == False and Q == False and R == False:
        print(F)
        return 'No agreementforest'
    
    
    
    
    
    
def MAFGenerator(Tree1,Tree2,k,cherry):
    global counter
    counter= 0
    MAF([Tree1],[Tree2],k,cherry)
    return counter
    
    
    
    
    
def FinalData(Data):
    obj = 0
    FinalData = list(Data)
    for i in range(0,len(Data)):
        tracker = MAFGenerator(Data[i][0][0][0],Data[i][0][0][1],15,list(Data[i][0][1]))
        FinalData[i] = FinalData[i].tolist()
        FinalData[i].append(tracker)
        obj +=1
        print(obj)
    return FinalData

New_Data2 = FinalData(Data)


np.save('preYNData2',New_Data2)
