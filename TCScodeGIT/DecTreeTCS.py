import ete3
import pandas as pd
import numpy as np
import random as rn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from sklearn.utils import resample
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pickle

"""
The Labelled data holds all the information
The used data has been adapted to work with the machine learning program
"""



Data = np.load(r'C:\Users\bryan\Desktop\master thesis\TCSProgram\CleanedLabeledData.npy',allow_pickle = True)
lijst = list()

for i in range(len(Data)):
    if None in Data[i]:
        del Data[i][7]
    Data[i] = np.array(Data[i])
    


for i in Data:
    lijst.append(np.array(i))
    
Data = np.array(lijst)
df = pd.DataFrame(data=Data, columns =["Cherry","feat1","feat2","feat3","feat4","size","TCS","classes"])



#col_idxX = np.array([1,2,3,4])
#X = Data[:,col_idxX]
#col_idxy = np.array([-1])
#y = Data[:,col_idxy]

df_majority = df[df.classes==0]
df_minority = df[df.classes==1]
df_minority_upsampled = resample(df_minority,replace=True,n_samples=988,random_state=123)
df_upsampled = pd.concat([df_majority,df_minority_upsampled])


X = df_upsampled.drop(columns=['Cherry','size','TCS','classes'],axis = 1)
y = df_upsampled.classes
y = y.astype('int')






X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)
y_train = y_train.astype('int')
y_test = y_test.astype('int')

TCStree = DecisionTreeClassifier(criterion = 'gini', max_depth = 9, random_state=0)
TCSforest = RandomForestClassifier(criterion = 'entropy', max_depth =11)


crossscore = cross_val_score(TCStree, X_train, y_train, cv=10)


TCStree.fit(X_train, y_train)
TCSforest.fit(X_train,y_train)



#export_graphviz(TCStree, out_file='TCS.dot',feature_names=['#TreesCherry', '#DifferenceCherry','Av Dist','Av LCAL'])
#
#
#filename = 'TCSDecTree.sav'
#
#pickle.dump(TCStree,open(filename,'wb'))

pred = TCStree.predict(X_test)
correct = 0
Fpos    = 0
Fneg    = 0

for i in range(len(pred)):
    if pred[i] == list(y_test)[i]:
        correct +=1
    if pred[i] ==1 and list(y_test)[i] == 0:
       Fpos +=1
    if pred[i] ==0 and list(y_test)[i] == 1:
        Fneg += 1

proby = TCStree.predict_proba(X_test)
proby = [p[1] for p in proby]

score = TCStree.score(X_test,y_test)
score2 = TCSforest.score(X_test,y_test)
aucscore = roc_auc_score(y_test,proby)

roc = sklearn.metrics.roc_curve(y_test, proby)
x = test[0]
y = test[1]

plt.subplots()
plt.plot(x,y, label = "ROC curve")
plt.plot([0,1],[0,1], label = "Base linear curve")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve TCS problem")
plt.legend()
plt.show()