import ete3
import numpy as np 
import random as rn
import math


Data = np.load(r'C:\Users\bryan\Desktop\master thesis\preYNData2.npy', allow_pickle=True)


"""
We want to group the data so that we can determine which cherries are right to pick and which are not.
"""


storage = []

i=0
k=i+1
while k < len(Data):
    k = i+1
    inner_storage=[]
    inner_storage.append(Data[i])
    while k < len(Data) and Data[i][0][0].all() == Data[k][0][0].all():
        inner_storage.append(Data[k])
        k+=1
    i += Data[i][1]
    storage.append(inner_storage)

for i in range(0,len(storage)):
   storage[i].sort(key=lambda x: x[6])

minimum_lijst = []

for j in range(0,len(storage)):
    minimum = []
    for f in range(0,len(storage[j])):
        minimum.append(storage[j][f][6])
    minimum_lijst.append(min(minimum))



for u in range(0,len(storage)):    
    for s in range(0,len(storage[u])):
#        if storage[u][s][6] == minimum_lijst[u]:
        if s <= math.ceil(len(storage[u])*0.5)-1:
            storage[u][s] = storage[u][s].tolist()
            storage[u][s].append(1)
        else:
            storage[u][s] = storage[u][s].tolist()
            storage[u][s].append(0)
            
Final_Data3 = []

for i in range(0,len(storage)):
    for j in range(0,len(storage[i])):
        Final_Data3.append(storage[i][j])

np.save('Final_Data3',Final_Data3)