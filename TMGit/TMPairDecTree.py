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



The_Data = np.load(r'C:\Users\bryan\Desktop\master thesis\TMPairTrainingData.npy',allow_pickle = True) #1046
The_Data2 = np.load(r'C:\Users\bryan\Desktop\master thesis\PairDataWithFeatures7Jul.npy',allow_pickle = True) #275 depth5
The_Data3 = np.load(r'C:\Users\bryan\Desktop\master thesis\PairDataWithFeatures9Jul.npy',allow_pickle = True) #1841
The_Data4 = np.load(r'C:\Users\bryan\Desktop\master thesis\PairDataWithFeatures14JulCase3.npy',allow_pickle = True) # 1568
The_Data5 = np.load(r'C:\Users\bryan\Desktop\master thesis\PairDataWithFeatures16JulTree.npy',allow_pickle = True) # 1464
The_Data6 = np.load(r'C:\Users\bryan\Desktop\master thesis\PairDataWithFeatures16JulRetic.npy',allow_pickle = True) # 3736
The_Data7 = np.load(r'C:\Users\bryan\Desktop\master thesis\PairDataFeatTest20Jul.npy',allow_pickle = True) # 2607
The_Data8 = np.load(r'C:\Users\bryan\Desktop\master thesis\PairDataWithFeatures26Jultest.npy',allow_pickle = True) # 3246
The_Data9 =  np.load(r'C:\Users\bryan\Desktop\master thesis\FinalTest10aug.npy',allow_pickle = True) # 1985

Data = The_Data9

#for i in range(len(The_Data)):
#    for j in The_Data[i]:
#        Data.append(j)

df = pd.DataFrame.from_records(data=Data, columns =["Node","Lemma","feat1","feat2","feat3","feat4","feat5","feat6","classes"])  #"feat4","feat5",'feat6',

df_majority = df[df.classes==0]
df_minority = df[df.classes==1]
df_minority_upsampled = resample(df_minority,replace=True,n_samples=1985,random_state=123)
df_upsampled = pd.concat([df_majority,df_minority_upsampled])


X = df_upsampled.drop(columns =["Node","Lemma","classes"], axis=1)
y = df_upsampled.classes



#Creating the test and training data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)
y_train = y_train.astype('int')
y_test = y_test.astype('int')


#Setup the initial tree/forest
TMPairtree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 12, random_state=0)
TMPairforest = RandomForestClassifier(criterion = 'entropy', max_depth =11)



#Performing cross-validation
crossscore = cross_val_score(TMPairtree, X_train, y_train, cv=10)


#Training the initial tree
TMPairtree.fit(X_train, y_train)
TMPairforest.fit(X_train,y_train)


#Evaluating the model
pred = TMPairtree.predict(X_test)
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

probs = TMPairtree.predict_proba(X_test)
proby = [p[1] for p in probs]

score = TMPairtree.score(X_test,y_test)
score2 = TMPairforest.score(X_test,y_test)
aucscore = roc_auc_score(y_test,proby)

roc = sklearn.metrics.roc_curve(y_test, proby)
#x = test[0]
#y = test[1]


#export_graphviz(TMPairtree, out_file='TMPair.dot',feature_names=['feat1', 'feat2','feat3','feat4','feat5'])


#filename = 'TMPairDecTreeFinal10aug.sav'
##
#pickle.dump(TMPairtree,open(filename,'wb'))
#
#x = roc[0]
#y = roc[1]
#
#plt.subplots()
#plt.plot(x,y, label = "ROC curve")
#plt.plot([0,1],[0,1], label = "Base linear curve")
#plt.xlabel("False positive rate")
#plt.ylabel("True positive rate")
#plt.title("ROC curve tail move problem")
#plt.legend()
#plt.show()


