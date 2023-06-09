# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 00:00:53 2023

@author: Vinay Sai
"""

import pandas as pd
import numpy as np

train = pd.read_csv("SalaryData_Train.csv")
train
train.dtypes

x_train_cont = train.select_dtypes("int" or "float")
x_train_cont

x_train_cat = train.select_dtypes("object")
x_train_cat

from sklearn.preprocessing import LabelEncoder,StandardScaler
LE = LabelEncoder()
for i in range(0,9):
    x_train_cat.iloc[:,i] = LE.fit_transform(x_train_cat.iloc[:,i])
x_train_cat    


SS = StandardScaler()
x_train_cont = SS.fit_transform(x_train_cont)    
x_train_cont = pd.DataFrame(x_train_cont)

x_train = pd.concat([x_train_cat.iloc[:,0:8],x_train_cont],axis=1)    
y_train = x_train_cat["Salary"]   

#######################################################################

test = pd.read_csv("SalaryData_Test.csv")
test

x_test_cont = test.select_dtypes("int" or "float")
x_test_cont

x_test_cat = test.select_dtypes("object")
x_test_cat

for i in range(0,9):
    x_test_cat.iloc[:,i] = LE.fit_transform(x_test_cat.iloc[:,i])
x_test_cat    


x_test_cont = SS.fit_transform(x_test_cont)    
x_test_cont = pd.DataFrame(x_test_cont)

x_test = pd.concat([x_test_cat.iloc[:,0:8],x_test_cont],axis=1)    
y_test = x_test_cat["Salary"]   


from sklearn.svm import SVC

# LINEAR
clf = SVC(kernel="linear",C=3)

clf.fit(x_train,y_train)

y_pred_train = clf.predict(x_train)
y_pred_test = clf.predict(x_test)

from sklearn.metrics import accuracy_score

print("accuracy_score of training is : ",accuracy_score(y_train, y_pred_train).round(4)) # accuracy_score of training is :  0.8106
print("accuracy_score of testing is : ",accuracy_score(y_test, y_pred_test).round(4))    # accuracy_score of testing is :  0.8091



# POLYNOMIAL

clf1 = SVC(kernel="poly",degree=5)

clf1.fit(x_train, y_train)

y_pred_train1 = clf1.predict(x_train)
y_pred_test1 = clf1.predict(x_test)


print("accuracy_score of training is : ",accuracy_score(y_train, y_pred_train1).round(4)) # accuracy_score of training is :  0.8265
print("accuracy_score of testing is : ",accuracy_score(y_test, y_pred_test1).round(4))    # accuracy_score of testing is :  0.8258

# RBF

clf2 = SVC(kernel="rbf",gamma=2)

clf2.fit(x_train,y_train)

y_pred_train2 = clf2.predict(x_train)
y_pred_test2 = clf2.predict(x_test)


print("accuracy_score of training is : ",accuracy_score(y_train, y_pred_train2).round(4)) #   accuracy_score of training is :  0.9246
print("accuracy_score of testing is : ",accuracy_score(y_test, y_pred_test2).round(4)) #   accuracy_score of testing is :  0.8063

