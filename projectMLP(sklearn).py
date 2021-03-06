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
from sklearn.neural_network import MLPClassifier
cls_mlp=MLPClassifier(hidden_layer_sizes=(10,10))
cls_mlp.fit(train_set,train_label)
pre_train=cls_mlp.predict(test_set)
d=accuracy_score(test_label,pre_train)
t2=time.time()
print("accuaracy :",d)
print( "training time:" ,t2-t1)





