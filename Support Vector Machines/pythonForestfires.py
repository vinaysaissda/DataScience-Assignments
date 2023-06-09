# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 19:17:28 2023

@author: Vinay Sai
"""

import pandas as pd
import numpy as np

df = pd.read_csv("forestfires.csv")
df

df.dtypes
df.shape

cat = df.select_dtypes("object")
contI = df.select_dtypes("int")
contF = df.select_dtypes("float")
contI
contF
cat
df

cont = pd.concat([contI,contF],axis=1)
cont
cont.dtypes

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
for i in range(0,3):
    cat.iloc[:,i] = LE.fit_transform(cat.iloc[:,i])
    
cat

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()

cont = SS.fit_transform(cont)
cont = pd.DataFrame(cont)
cont

y= cat["size_category"]
x= pd.concat([cont,cat.iloc[:,0:2]],axis=1)
x


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.3,random_state=(5))

from sklearn.svm import SVC

# LINEAR
clf = SVC(kernel="linear",C=3)

clf.fit(x_train,y_train)

y_pred_train = clf.predict(x_train)
y_pred_test = clf.predict(x_test)

from sklearn.metrics import mean_squared_error

print("rmse of training is : ",np.sqrt(mean_squared_error(y_train, y_pred_train)).round(4)) # rmse of training is :  0.2664
print("rmse of testing is : ",np.sqrt(mean_squared_error(y_test, y_pred_test)).round(4))    # rmse of testing is :  0.3154


# POLYNOMIAL

clf1 = SVC(kernel="poly",degree=5)

clf1.fit(x_train, y_train)

y_pred_train1 = clf1.predict(x_train)
y_pred_test1 = clf1.predict(x_test)


print("rmse of training is : ",np.sqrt(mean_squared_error(y_train, y_pred_train1)).round(4)) # rmse of training is :  0.4752
print("rmse of testing is : ",np.sqrt(mean_squared_error(y_test, y_pred_test1)).round(4))    # rmse of testing is :  0.4817

# RBF

clf2 = SVC(kernel="rbf",gamma=3)

clf2.fit(x_train,y_train)

y_pred_train2 = clf2.predict(x_train)
y_pred_test2 = clf2.predict(x_test)


print("rmse of training is : ",np.sqrt(mean_squared_error(y_train, y_pred_train2)).round(4)) # rmse of training is :  0.1136
print("rmse of testing is : ",np.sqrt(mean_squared_error(y_test, y_pred_test2)).round(4)) #   rmse of testing is :  0.4874
