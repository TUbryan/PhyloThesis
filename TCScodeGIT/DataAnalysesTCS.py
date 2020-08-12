import numpy as np
import matplotlib.pyplot as plt
import copy


#Data = np.load(r'C:\Users\bryan\Desktop\master thesis\spannendetestdata.npy', allow_pickle=True) #before 18 may
#Data2 = np.load(r'C:\Users\bryan\Desktop\master thesis\spannendetestdata2.npy', allow_pickle=True)
#Data3 = np.load(r'C:\Users\bryan\Desktop\master thesis\superTCSData.npy', allow_pickle=True)

Data2 = np.load(r'C:\Users\bryan\Desktop\master thesis\TCSProgram\2retResults.npy', allow_pickle=True)
Data3 = np.load(r'C:\Users\bryan\Desktop\master thesis\TCSProgram\3retResults.npy', allow_pickle=True)
Data4 = np.load(r'C:\Users\bryan\Desktop\master thesis\TCSProgram\4retResults.npy', allow_pickle=True)
Data5 = np.load(r'C:\Users\bryan\Desktop\master thesis\TCSProgram\5retResults.npy', allow_pickle=True)
Data6 = np.load(r'C:\Users\bryan\Desktop\master thesis\TCSProgram\6retResults.npy', allow_pickle=True)
Data7 = np.load(r'C:\Users\bryan\Desktop\master thesis\TCSProgram\7retResults.npy', allow_pickle=True)


"""
Graph creator part
"""

#x = range(0,10)
#y1 = []
#y2 = []
#y3 = []
#y4 = []
#
#
##for i in range(0,25):
##    if Data[0][i][0][0] == None:
##        Data[0][i][0][2] = 1000
##        
##for i in range(0,25):
##    if Data2[0][i][0][0] == None:
##        Data2[0][i][0][2] = 1000
##        
##for i in range(0,50):
##    if Data3[0][i][0][0] == None:
##        Data3[0][i][0][2] = 1000
##    if Data3[1][i][0][0] == None:
##        Data3[1][i][0][2] = 1000
##
#for i in range(0,10):
#    y1.append(Data[0][i][0][2])
#    y2.append(Data[1][i][0][2])
#
#for i in range(0,50):
#    y3.append(Data3[0][i][0][1])
#    y4.append(Data3[1][i][0][1])


#plt.subplot()
#fig = plt.figure(figsize=(20,5))
#ax = fig.add_subplot(111)
#ax.scatter(x, y1, s=10 , c='blue', label = 'regular algorithm')
#ax.scatter(x, y2, s=5 , c='red', label = 'Algorithm with ML')
#plt.title('Runtime 7 Reticulations')
#ax.legend()
#
#plt.show()


"""
Generating data for the table. Here we will display for each reticulation number 2-7
how many instances were solved within 10 minutes, the average runtime for the #reticulations,
average size of the search tree for solved instances and the average size search tree for all unsolved
instanced per reticulation. This will hopefully give an idea on how the new algorithm performs
against the old algorithm
any list will have:

[# of solved instances, average runtime solved instances, av size search tree solved instances,
average size search tree unsolved instances]
for both the reg alg and the ml alg. Giving a total of 8 instances per list

"""


ret2 = []
ret3 = []
ret4 = []
ret5 = []
ret6 = []
ret7 = []

for j in range(0,1):
    tracker1 = 0
    tracker2 = 0
    RT1 = 0
    RT2 = 0
    SST1 = 0
    SST2 = 0
    UST1 = 0
    UST2 = 0
    for i in range(len(Data2[0])):
        if Data2[0][i][0][0] != None:
            tracker1 += 1
            RT1 += Data2[0][i][0][2]
            SST1 += Data2[0][i][0][1]
        if Data2[0][i][0][0] == None:
            UST1 += Data2[0][i][0][1]
        if Data2[1][i][0][0] != None:
            tracker2 +=1
            RT2 += Data2[1][i][0][2]
            SST2 += Data2[1][i][0][1]
        if Data2[1][i][0][0] == None:
            UST2 += Data2[1][i][0][1]
    ret2.append(tracker1)
    ret2.append(tracker2)
    try:
       ret2.append(RT1/tracker1)
    except:
       ret2.append(0)
    try:    
       ret2.append(RT2/tracker2)
    except:
       ret2.append(0)
    try:
       ret2.append(SST1/tracker1)
    except:
       ret2.append(0)
    try:
       ret2.append(SST2/tracker2)
    except:
       ret2.append(0)
    try:
       ret2.append(UST1/(10-tracker1))
    except:
       ret2.append(0)
    try:
       ret2.append(UST2/(10-tracker2))
    except:
       ret2.append(0)
       
       
for j in range(0,1):
    tracker1 = 0
    tracker2 = 0
    RT1 = 0
    RT2 = 0
    SST1 = 0
    SST2 = 0
    UST1 = 0
    UST2 = 0
    for i in range(len(Data3[0])):
        if Data3[0][i][0][0] != None:
            tracker1 += 1
            RT1 += Data3[0][i][0][2]
            SST1 += Data3[0][i][0][1]
        if Data3[0][i][0][0] == None:
            UST1 += Data3[0][i][0][1]
        if Data3[1][i][0][0] != None:
            tracker2 +=1
            RT2 += Data3[1][i][0][2]
            SST2 += Data3[1][i][0][1]
        if Data3[1][i][0][0] == None:
            UST2 += Data3[1][i][0][1]
    ret3.append(tracker1)
    ret3.append(tracker2)
    try:
       ret3.append(RT1/tracker1)
    except:
       ret3.append(0)
    try:    
       ret3.append(RT2/tracker2)
    except:
       ret3.append(0)
    try:
       ret3.append(SST1/tracker1)
    except:
       ret3.append(0)
    try:
       ret3.append(SST2/tracker2)
    except:
       ret3.append(0)
    try:
       ret3.append(UST1/(10-tracker1))
    except:
       ret3.append(0)
    try:
       ret3.append(UST2/(10-tracker2))
    except:
       ret3.append(0)

for j in range(0,1):
    tracker1 = 0
    tracker2 = 0
    RT1 = 0
    RT2 = 0
    SST1 = 0
    SST2 = 0
    UST1 = 0
    UST2 = 0
    for i in range(len(Data4[0])):
        if Data4[0][i][0][0] != None:
            tracker1 += 1
            RT1 += Data4[0][i][0][2]
            SST1 += Data4[0][i][0][1]
        if Data4[0][i][0][0] == None:
            UST1 += Data4[0][i][0][1]
        if Data4[1][i][0][0] != None:
            tracker2 +=1
            RT2 += Data4[1][i][0][2]
            SST2 += Data4[1][i][0][1]
        if Data4[1][i][0][0] == None:
            UST2 += Data4[1][i][0][1]
    ret4.append(tracker1)
    ret4.append(tracker2)
    try:
       ret4.append(RT1/tracker1)
    except:
       ret4.append(0)
    try:    
       ret4.append(RT2/tracker2)
    except:
       ret4.append(0)
    try:
       ret4.append(SST1/tracker1)
    except:
       ret4.append(0)
    try:
       ret4.append(SST2/tracker2)
    except:
       ret4.append(0)
    try:
       ret4.append(UST1/(10-tracker1))
    except:
       ret4.append(0)
    try:
       ret4.append(UST2/(10-tracker2))
    except:
       ret4.append(0)
       
for j in range(0,1):
    tracker1 = 0
    tracker2 = 0
    RT1 = 0
    RT2 = 0
    SST1 = 0
    SST2 = 0
    UST1 = 0
    UST2 = 0
    for i in range(len(Data5[0])):
        if Data5[0][i][0][0] != None:
            tracker1 += 1
            RT1 += Data5[0][i][0][2]
            SST1 += Data5[0][i][0][1]
        if Data5[0][i][0][0] == None:
            UST1 += Data5[0][i][0][1]
        if Data5[1][i][0][0] != None:
            tracker2 +=1
            RT2 += Data5[1][i][0][2]
            SST2 += Data5[1][i][0][1]
        if Data5[1][i][0][0] == None:
            UST2 += Data5[1][i][0][1]
    ret5.append(tracker1)
    ret5.append(tracker2)
    try:
       ret5.append(RT1/tracker1)
    except:
       ret5.append(0)
    try:    
       ret5.append(RT2/tracker2)
    except:
       ret5.append(0)
    try:
       ret5.append(SST1/tracker1)
    except:
       ret5.append(0)
    try:
       ret5.append(SST2/tracker2)
    except:
       ret5.append(0)
    try:
       ret5.append(UST1/(10-tracker1))
    except:
       ret5.append(0)
    try:
       ret5.append(UST2/(10-tracker2))
    except:
       ret5.append(0)

for j in range(0,1):
    tracker1 = 0
    tracker2 = 0
    RT1 = 0
    RT2 = 0
    SST1 = 0
    SST2 = 0
    UST1 = 0
    UST2 = 0
    for i in range(len(Data6[0])):
        if Data6[0][i][0][0] != None:
            tracker1 += 1
            RT1 += Data6[0][i][0][2]
            SST1 += Data6[0][i][0][1]
        if Data6[0][i][0][0] == None:
            UST1 += Data6[0][i][0][1]
        if Data6[1][i][0][0] != None:
            tracker2 +=1
            RT2 += Data6[1][i][0][2]
            SST2 += Data6[1][i][0][1]
        if Data6[1][i][0][0] == None:
            UST2 += Data6[1][i][0][1]
    ret6.append(tracker1)
    ret6.append(tracker2)
    try:
       ret6.append(RT1/tracker1)
    except:
       ret6.append(0)
    try:    
       ret6.append(RT2/tracker2)
    except:
       ret6.append(0)
    try:
       ret6.append(SST1/tracker1)
    except:
       ret6.append(0)
    try:
       ret6.append(SST2/tracker2)
    except:
       ret6.append(0)
    try:
       ret6.append(UST1/(10-tracker1))
    except:
       ret6.append(0)
    try:
       ret6.append(UST2/(10-tracker2))
    except:
       ret6.append(0)
       
for j in range(0,1):
    tracker1 = 0
    tracker2 = 0
    RT1 = 0
    RT2 = 0
    SST1 = 0
    SST2 = 0
    UST1 = 0
    UST2 = 0
    for i in range(len(Data7[0])):
        if Data7[0][i][0][0] != None:
            tracker1 += 1
            RT1 += Data7[0][i][0][2]
            SST1 += Data7[0][i][0][1]
        if Data7[0][i][0][0] == None:
            UST1 += Data7[0][i][0][1]
        if Data7[1][i][0][0] != None:
            tracker2 +=1
            RT2 += Data7[1][i][0][2]
            SST2 += Data7[1][i][0][1]
        if Data7[1][i][0][0] == None:
            UST2 += Data7[1][i][0][1]
    ret7.append(tracker1)
    ret7.append(tracker2)
    try:
       ret7.append(RT1/tracker1)
    except:
       ret7.append(0)
    try:    
       ret7.append(RT2/tracker2)
    except:
       ret7.append(0)
    try:
       ret7.append(SST1/tracker1)
    except:
       ret7.append(0)
    try:
       ret7.append(SST2/tracker2)
    except:
       ret7.append(0)
    try:
       ret7.append(UST1/(10-tracker1))
    except:
       ret7.append(0)
    try:
       ret7.append(UST2/(10-tracker2))
    except:
       ret7.append(0)
       
for k in range(0,8):
    ret2[k] = round(ret2[k])
    ret3[k] = round(ret3[k])
    ret4[k] = round(ret4[k])
    ret5[k] = round(ret5[k])
    ret6[k] = round(ret6[k])
    ret7[k] = round(ret7[k])
       
       
       
       
       
       
       
       
       
       
       
       
    

