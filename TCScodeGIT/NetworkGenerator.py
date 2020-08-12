"""
Here we create a program that can generate a network and from that network
generates t or fewer trees to form our data set.
"""

import numpy as np
import ete3
import random as rn
import networkx as nx



#We will now create a function that can generate a network on n leaves with k reticulations

def NetworkGenerator(n,k):
    #We first set limiting parameters
    N = nx.DiGraph()
    s = n+k-2
    kr = k
    N.add_node(0)
    N.add_edge(0,1)
    N.add_edge(0,2)
    w=3
    #We will add either a reticulation or a tree node to the network
    while s >0 and kr >0:
        decision = np.random.choice(['ret','tree'],p=[kr/(s+kr),s/(s+kr)])
        if decision == 'ret':
            Leafset = [n for n in N.nodes() if N.out_degree(n) == 0]
            M = []
            for i in Leafset:
                if (N.in_degree([n for n in N.predecessors(i)][0]) == 1 or N.in_degree([n for n in N.predecessors(i)][0]) == 0) and N.out_degree([n for n in N.predecessors(i)][0]) == 2:
                    for j in N.successors([n for n in N.predecessors(i)][0]):
                        if j == i:
                            continue
                        elif N.in_degree(j) == 1 and N.out_degree(j) == 2:
                            M.append(i)
                        elif N.out_degree(j) == 0 and N.in_degree(j) == 1:
                           M.append(i)
            M = list(dict.fromkeys(M))
            if (len(M) == 1) or (len(M) ==2 and [n for n in N.predecessors(M[0])] == [n for n in N.predecessors(M[1])]):
                u = rn.choice(Leafset)
                N.add_edge(u,w)
                w +=1
                N.add_edge(u,w)
                w +=1
                s = s-1
            else:
                sample = rn.sample(M,2)
                while [n for n in N.predecessors(sample[0])] == [l for l in N.predecessors(sample[1])]:
                    sample = rn.sample(M,2)
                u = sample[0]
                v = sample[1]
                pred = [n for n in N.predecessors(v)]
                N.add_edge(pred[0],u)
                N.remove_node(v)
                N.add_edge(u,w)
                w +=1
                kr = kr-1
        if decision == 'tree':
            Leafset = [n for n in N.nodes() if N.out_degree(n) == 0]
            u = rn.choice(Leafset)
            N.add_edge(u,w)
            w +=1
            N.add_edge(u,w)
            w +=1
            s = s-1
    return N
 

#Here we generate the trees from the network.           
def TreeSetGenerator(N,t):
    t = t
    N = N
    tracker = 0
    trees = []
    treesedges = []
    while len(trees) < t and tracker < 100:
        F = N.copy()
        reticulations = [n for n in F.nodes() if (F.in_degree(n) == 2 and F.out_degree(n) == 1)]
        for i in reticulations:
            predecessors = [n for n in F.predecessors(i)]
            if 0 in predecessors:
                predecessors.remove(0)
            edge = rn.choice(predecessors)
            F.remove_edge(edge,i)
            Suppress_node_set = [n for n in F.nodes() if (F.in_degree(n) ==1 and F.out_degree(n) == 1)]
            for k in Suppress_node_set:
                pred = [n for n in F.predecessors(k)][0]
                suc  = [n for n in F.successors(k)][0]
                F.add_edge(pred,suc)
                F.remove_node(k)
        if trees == []:
            trees.append(F)
            treesedges.append(F.edges())
        if len(trees) >= 1:
            if F.edges() in treesedges:
                tracker +=1
            else:
                trees.append(F)
    return trees
    


def tree_to_newick(tree,key):
    global keylist1
    global keylist2
    if len(tree[key].keys())>0:
        keys = list(tree[key].keys())
        subt1 = tree_to_newick(tree,keys[0])
        subt2 = tree_to_newick(tree,keys[1])
        subt = (subt1,subt2)
        keylist1.append(str(subt))
        keylist2.append(str(subt) +str(key))
        return subt 
    if len(tree[key].keys()) == 0:
        return key

def final_tree_to_newick(newick):
    global keylist1
    global keylist2
    keylist1 = keylist1[::-1]
    keylist2 = keylist2[::-1]
    newick = str(newick)
    for i in range(len(keylist1)):
        newick = newick.replace(keylist1[i],keylist2[i])
    tree = "".join(newick)+';'
    return tree

def cleanupnewick(subt,key):
    return
       



def TreeSetConverter(trees):
    global keylist1
    global keylist2
    treeset = []
    for i in trees:
        keylist1 = []
        keylist2 = []
        dictionary = nx.to_dict_of_dicts(i)
        raw_newick = tree_to_newick(dictionary,0)
        newick = final_tree_to_newick(raw_newick)
        tree = ete3.Tree(newick)
        treeset.append(tree)
    return treeset

def DataGenerator():
    Data = []
    for i in range(0,200):
       try:
          Subdata = []
          reticulation = rn.randint(3,6)
          Leaves = rn.randint(15,20)
          Network = NetworkGenerator(Leaves,reticulation)
          trees = rn.randint(4,10)
          Trees = TreeSetGenerator(Network,trees)
          TreeNewick = TreeSetConverter(Trees)
          Subdata = [reticulation,TreeNewick]
          Data.append(Subdata)
       except:
           pass
    return Data
       
       
"""
For the training data we used 2 to 7 retics and 25 to 35 leaves and 2 to 20 trees.
For first analyses we used 2 to 3 rets and 10 to 15 leaves and 2 to 10 trees.
"""   



#test = NetworkGenerator(20,reticulations) 
#test2 = TreeSetGenerator(test,3)
#Data = TreeSetConverter(test2)
#Data = [ete3.Tree(i,format = 8) for i in Data]
#Final_Data = [reticulations,Data]
#np.save('Data',Final_Data)
#
test = DataGenerator()