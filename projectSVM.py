import pandas as pd
import time
import numpy as np
import pickle as pkl
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

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
cls_svm=SVC(kernel='linear',gamma='auto')
cls_svm.fit(train_set,train_label)
pre_train=cls_svm.predict(test_set)
c=accuracy_score(test_label,pre_train)
t2=time.time()
print("accuracy:",c)
print("training time:",t2-t1)





