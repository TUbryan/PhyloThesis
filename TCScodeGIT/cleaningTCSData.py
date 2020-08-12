import numpy as np

Data = np.load(r"C:\Users\bryan\Desktop\master thesis\LabeledData.npy", allow_pickle =True)

CleanedList = []


for i in range(len(Data)):
    for j in range(len(Data[i][0][1])):
        sublist = []
        for k in Data[i][0][1][j][0]:
            sublist.append(k)
        for k in Data[i][0][1][j][1:]:
            sublist.append(k)
        CleanedList.append(sublist)