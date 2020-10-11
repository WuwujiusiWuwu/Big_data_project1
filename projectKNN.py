import pandas as pd
import time
import numpy as np
from sklearn.metrics import accuracy_score

#import data
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
t1=time.time()
from sklearn.neighbors import KNeighborsClassifier
cls_KNN = KNeighborsClassifier(n_neighbors=13, weights='uniform', algorithm='auto', leaf_size=30)
cls_KNN.fit(train_set,train_label)
pre_train=cls_KNN.predict(test_set)
a=accuracy_score(test_label,pre_train)
t2=time.time()
print("accuracy:",a)
print("training time:",t2-t1)






