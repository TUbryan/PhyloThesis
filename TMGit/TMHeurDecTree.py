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



The_Data = np.load(r'C:\Users\bryan\Desktop\master thesis\HeurImpr2.npy',allow_pickle = True)

Data = []

for i in range(len(The_Data)):
    for j in The_Data[i]:
        Data.append(j)

df = pd.DataFrame.from_records(data=Data, columns =["Node","Network","target","feat1","feat2","feat3","feat4","Lemma","classes"])

X = df.drop(columns =["Node","Network","target","Lemma","classes"], axis=1)
y=df.classes



#Creating the test and training data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)
y_train = y_train.astype('int')
y_test = y_test.astype('int')


#Setup the initial tree/forest
TMHeurtree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 10, random_state=0)
TMHeurforest = RandomForestClassifier(criterion = 'entropy', max_depth =11)



#Performing cross-validation
crossscore = cross_val_score(TMHeurtree, X_train, y_train, cv=10)


#Training the initial tree
TMHeurtree.fit(X_train, y_train)
TMHeurforest.fit(X_train,y_train)


#Evaluating the model
pred = TMHeurtree.predict(X_test)
correct = 0
Fpos    = 0
Fneg    = 0

#for i in range(len(pred)):
#    if pred[i] == list(y_test)[i]:
#        correct +=1
#    if pred[i] ==1 and list(y_test)[i] == 0:
#       Fpos +=1
#    if pred[i] ==0 and list(y_test)[i] == 1:
#        Fneg += 1

proby = TMHeurtree.predict_proba(X_test)
proby = [p[1] for p in proby]

score = TMHeurtree.score(X_test,y_test)
score2 = TMHeurforest.score(X_test,y_test)
aucscore = roc_auc_score(y_test,proby)

roc = sklearn.metrics.roc_curve(y_test, proby)
#x = test[0]
#y = test[1]


#export_graphviz(TMHeurtree, out_file='TMHeur.dot',feature_names=['feat1', 'feat2','feat3','feat4'])
#

filename = 'TMHeurDecTree.sav'

pickle.dump(TMHeurtree,open(filename,'wb'))