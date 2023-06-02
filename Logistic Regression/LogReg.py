# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 13:08:26 2023

@author: Vinay Sai
"""

import pandas as pd
import numpy as np

df = pd.read_csv("bank-full.csv",sep=";")
df
df.shape
df.dtypes

cat = df.select_dtypes("object")
cont = df.select_dtypes("int")
cat
cont

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
for i in range(0,10):
    cat.iloc[:,i] = LE.fit_transform(cat.iloc[:,i])
cat

cat.columns
y = cat["y"]
x = pd.concat([cont,cat.iloc[:,:9]],axis=1)
x

from sklearn.linear_model import LogisticRegression
LogR = LogisticRegression()
LogR.fit(x, y)


y_pred = LogR.predict(x)
y_pred
y_proba = LogR.predict_log_proba(x)[:,1]


from sklearn.metrics import accuracy_score,recall_score,f1_score,confusion_matrix,precision_score,roc_auc_score,roc_curve
fpr,tpr,_ = roc_curve(y, y_proba)

print("confusion matrix :",confusion_matrix(y, y_pred))
print("accuracy score :",accuracy_score(y, y_pred))
print("recall score is ",recall_score(y, y_pred))
print("precision score :",precision_score(y, y_pred))
print("f1 score is ",f1_score(y, y_pred))
print("Area under curve :",roc_auc_score(y, y_proba))

import matplotlib.pyplot as plt
plt.scatter(fpr, tpr)
plt.plot(fpr, tpr,color= "red")
plt.show()

'''
EXPECTED OUTPUT :
    confusion matrix : [[39279   643]
                        [ 4437   852]]
    accuracy score : 0.8876379642122493
    recall score is  0.16108905275099264
    precision score : 0.5698996655518395
    f1 score is  0.2511792452830189
    Area under curve : 0.8112257501106169
'''
    