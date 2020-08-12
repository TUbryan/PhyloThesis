import ete3
import random as rn
import numpy as np


"""
This program will simulate random rSPR moves to help create testdata for our machine learning program
We do this by detaching and ataching random parts of the tree
"""

names = list(np.arange(0,10000,1))

#This setup function will return all the nodes of a tree in a neet list. It wil not take the root node, because we will never perform
#an rSPR move with the root node as a head node.

def Setup(T):
    node_list = []
    for i in T.traverse():
        node_list.append(i.name)
    return node_list[3:]



#This function simulates a random rSPR move within the tree.

def rSPR_move(Tree, string):
   F= Tree.copy()
   aux = ete3.Tree()
   aux.populate(0,names_library = [string])
   node_list = []
#   Tree.show()
   for i in Tree.traverse():
       node_list.append(i.name)
   edge = rn.choice(node_list[5:])
   edge = Tree.search_nodes(name = edge)[0]
   unavailable = ['0',edge.up.name]
   new_node_list = []
   for i in edge.traverse():
       unavailable.append(i.name)
   for i in node_list:
       if i not in unavailable:
          new_node_list.append(i)
#    new_node_list = Setup(Tree)
#    new_name = rn.choice(new_node_list)
   try:
      new_name = rn.choice(new_node_list)
   except:
      new_node_list.append(string)
      new_name =rn.choice(new_node_list)
#   except:
#       print('hier ben ik')
#       for i in Tree.traverse():
#           print(i.name)
#       print(F)
#       print(edge)
#       print(edge.name)
#       for i in edge.traverse():
#           print(i.name)
#       print(unavailable)
#       print(node_list)
#       print(new_node_list)
#       for i in Tree.traverse():
#           print(i.name)
   try:
      location = Tree.search_nodes(name = new_name)[0]
   except:
       print(Tree)
       print(edge)
       print(new_name)
   location.up.add_child(aux)
#   except:
#       print(edge)
#       print(new_node_list)
#       
#       print(location)
#       print(location.name)
#       print(new_name)
#       Tree.show()
#       print('hoi')
   edge.detach()
   for i in Tree.traverse():
       if len(i.get_children()) == 1:
           i.delete()
   location.detach()
   for i in Tree.traverse():
       if len(i.get_children()) == 1:
           i.delete()
   new_node = Tree.search_nodes(name = str(string))[0]
   new_node.add_child(location)
   new_node.add_child(edge)
   return Tree
    




#The input will be the nodes a tree in a list T and a parameter k that indicates the number of random rSPR moves that will be made.
#The output will be a tuple of the original tree and the randomized tree. 


def TreeRandomizer(T,k):
    F = T.copy()
    tracker = 0
    length = 315
    for i in T.traverse():
        length += 1
    while tracker < k:
        F = rSPR_move(F,str(length))
        tracker += 1
        length += 1
    return (T,F)


def TreeMaker(n): 
    t = ete3.Tree()
    t.populate(n, names_library = names)
    i = 0
    for n in t.traverse():
        if n.name == '':
            n.name = str(i)
        i +=1
    return t

#t = TreeMaker(10)
#test = []
#for i in range(0,1000):
#    test.append(TreeRandomizer(t,2))

   
   
   
   
   
   
   
   
   
   
   
   
   
   
   