#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
data1 = pd.read_csv("new_data.csv")
data1 = data1.iloc[:, 0:21]
data1 = np.array(data1)
train_label = data1[:,4]
train_set = data1[:,5:21]
data2 = pd.read_csv("test_set.csv")
data2 = data2.iloc[:, 0:21]
data2 = np.array(data2)
test_label = data2[:, 4]
test_set= data2[:,5:21]
SVM_clf = SVC(kernel='linear',gamma='auto',probability=True)
MLP_clf = MLPClassifier(hidden_layer_sizes=(10, 10))
DT_clf =tree.DecisionTreeClassifier(max_depth=7,min_samples_split=26)
KNN_clf=KNeighborsClassifier(n_neighbors=13, weights='uniform', algorithm='auto', leaf_size=30)
ensemble = VotingClassifier(estimators=[('mlp', MLP_clf), ('svm', SVM_clf),
                                        ('dt', DT_clf), ('knn', KNN_clf)], voting='soft')
ensemble.fit(train_set,train_label)
pre_train=ensemble.predict(test_set)
b=accuracy_score(test_label,pre_train)
print("accuracy:",b)


# In[ ]:




