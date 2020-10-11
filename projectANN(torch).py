import torch 
import time
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

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

X_train = torch.FloatTensor(train_set) 
X_test = torch.FloatTensor(test_set) 
y_train = torch.LongTensor(train_label) 
y_test = torch.LongTensor(test_label)
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=16, out_features=96)
        self.output = nn.Linear(in_features=96, out_features=3)
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.output(x)
        x = F.softmax(x,dim=1)
        return x

model = ANN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 100
loss_arr = []
t1=time.time()
for i in range(epochs):
    y_hat = model.forward(X_train)
    loss = criterion(y_hat, y_train)
    loss_arr.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
predict_out = model(X_test)
_,predict_y = torch.max(predict_out, 1)
predict_ypredict_out = model(X_test)
t2=time.time()
from sklearn.metrics import accuracy_score
print("accuracy :", accuracy_score(y_test, predict_y) )
print("training time:",t2-t1)





