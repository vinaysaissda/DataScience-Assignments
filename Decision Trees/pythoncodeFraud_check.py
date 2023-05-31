# -*- coding: utf-8 -*-
"""
@author: Vinay Sai
"""
import pandas as pd
import numpy as np

df = pd.read_csv("Fraud_check.csv")
df
 
for i in range(0,600):
    if df["Taxable.Income"][i] <= 30000:
        df["Taxable.Income"][i] = "NO"
    else:
        df["Taxable.Income"][i] = "YES"

df.dtypes
n=6
test1 = []
test2 = []
df_cat = pd.DataFrame(test1)

df_cont = pd.DataFrame(test2)
for i in range(0,n):
    if df.iloc[:,i].dtype == "int64" :
       df_cont = pd.concat([df_cont,df.iloc[:,i]],axis=1) 
    if df.iloc[:,i].dtype == "object":
       df_cat = pd.concat([df_cat,df.iloc[:,i]],axis=1)

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

df_cat
df.shape
df_cont
for i in range(0,4):
    df_cat.iloc[:,i] = LE.fit_transform(df_cat.iloc[:,i])
df_cat

y = df_cat['Taxable.Income']
x1 = df_cat[df_cat.columns[[0,1,3]]]
x1
df_cont
x = pd.concat([x1,df_cont],axis=1)

# Model fitting
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=(2),train_size=0.6)

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(max_depth=5)
DT.fit(x_train, y_train)

# Predict
y_pred_train =  DT.predict(x_train)
y_pred_test = DT.predict(x_test)
y_pred_train
y_pred_test

y_pred_train.shape
y_pred_test.shape


from sklearn.metrics import accuracy_score
print("accuracy train score : ",accuracy_score(y_train, y_pred_train)) #accuracy train score :  0.8166666666666667
print("accuracy test score : ",accuracy_score(y_test, y_pred_test))    #accuracy test score :  0.7875

from sklearn.tree import DecisionTreeRegressor
DT1 = DecisionTreeRegressor(max_depth=(5)) 
DT1.fit(x_train, y_train)

# Predict
y_pred_train =  DT1.predict(x_train)
y_pred_test = DT1.predict(x_test)
y_pred_train
y_pred_test

y_pred_train.shape
y_pred_test.shape

from sklearn.metrics import mean_squared_error

err1 = np.sqrt(mean_squared_error(y_train, y_pred_train))
err2 = np.sqrt(mean_squared_error(y_test, y_pred_test))
print("training rmse :",  err1)  #training rmse : 0.37256101185135776
print("testing rmse3 :",err2)    #testing rmse3 : 0.4255056774975852





