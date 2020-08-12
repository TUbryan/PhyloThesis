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



Data = np.load(r'C:\Users\bryan\Desktop\master thesis\TMData_with_new_feature.npy',allow_pickle = True)




df = pd.DataFrame.from_records(data=Data, columns =["Move","feat1","feat2","feat3","feat4","feat5","feat6","Lemma","classes"])

X = df.drop(columns =["Move","Lemma","classes"], axis=1)
y=df.classes



#Creating the test and training data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)
y_train = y_train.astype('int')
y_test = y_test.astype('int')


#Setup the initial tree/forest
TMtree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 30, random_state=0)
TMforest = RandomForestClassifier(criterion = 'entropy', max_depth =30)



#Performing cross-validation
crossscore = cross_val_score(TMtree, X_train, y_train, cv=10)


#Training the initial tree
TMtree.fit(X_train, y_train)
TMforest.fit(X_train,y_train)


#Evaluating the model
pred = TMtree.predict(X_test)
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

proby = TMtree.predict_proba(X_test)
proby = [p[1] for p in proby]

score = TMtree.score(X_test,y_test)
score2 = TMforest.score(X_test,y_test)
aucscore = roc_auc_score(y_test,proby)

roc = sklearn.metrics.roc_curve(y_test, proby)
#x = test[0]
#y = test[1]


#export_graphviz(TMtree, out_file='TM.dot',feature_names=['feat1', 'feat2','feat3','feat4','feat5','feat6'])
#

filename = 'TMDecTree.sav'

pickle.dump(TMtree,open(filename,'wb'))



