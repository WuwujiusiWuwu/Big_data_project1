import pandas as pd
import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree

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
cls_DT=tree.DecisionTreeClassifier(max_depth=7,min_samples_split=26)
cls_DT.fit(train_set,train_label)
pre_train=cls_DT.predict(test_set)
b=accuracy_score(test_label,pre_train)
t2=time.time()
print("accuracy:",b)
print("training time:",t2-t1)

#Optimization model parameters
#using cross validation to find the beat max_depth
def cv_score(d):
    clf = tree.DecisionTreeClassifier(max_depth=d)
    clf.fit(train_set, train_label)
    return(clf.score(train_set,train_label), clf.score(test_set, test_label))
depths = np.arange(1,10)
scores = [cv_score(d) for d in depths]
tr_scores = [s[0] for s in scores]
te_scores = [s[1] for s in scores]
# Find the index with the highest score for the cross-validation data set
tr_best_index = np.argmax(tr_scores)
te_best_index = np.argmax(te_scores)

print("bestdepth:", te_best_index+1, " bestdepth_score:", te_scores[te_best_index], '\n')

#using minsplit to optimize the model

def minsplit_score(val):
    clf = tree.DecisionTreeClassifier(criterion='gini',max_depth=7, min_impurity_decrease=val)
    clf.fit(train_set,train_label)
    return (clf.score(train_set,train_label), clf.score(test_set, test_label), )

# Specify parameter range, train model separately and calculate score

vals = np.linspace(0, 0.2, 100)
scores = [minsplit_score(v) for v in vals]
tr_scores = [s[0] for s in scores]
te_scores = [s[1] for s in scores]

bestmin_index = np.argmax(te_scores)
bestscore = te_scores[bestmin_index]
print("bestmin:", vals[bestmin_index])
print("bestscore:", bestscore)

#Using F1-score, recall and precision to measure the effect of classification of the DT classifier.
model = tree.DecisionTreeClassifier(max_depth=7, min_impurity_decrease=0.0)
model.fit(train_set, train_label)
from sklearn import metrics
print("tees_score:", model.score(test_set,test_label))

y_pred = model.predict(test_set)

print("precision:",metrics.precision_score(test_label, y_pred))
print("recall rate :",metrics.recall_score(test_label, y_pred))
print("F1_score:",metrics.f1_score(test_label, y_pred))



#use GridSearchCV to optimize the parameters
from sklearn.model_selection import GridSearchCV

entropy_thresholds = np.linspace(0, 1, 100)
gini_thresholds = np.linspace(0, 0.2, 100)
#Set the parameter matrix:
param_grid = [{'criterion': ['entropy'], 'min_impurity_decrease': entropy_thresholds},
              {'criterion': ['gini'], 'min_impurity_decrease': gini_thresholds},
              {'max_depth': np.arange(2,10)},
              {'min_samples_split': np.arange(2,30,2)}]
clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, cv=5)
clf.fit(train_set, train_label)
clf.fit(test_set,test_label)
print("best param:{0}\nbest score:{1}".format(clf.best_params_, clf.best_score_))






