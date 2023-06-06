# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 12:39:37 2023

@author: Vinay Sai
"""

import pandas as pd
import numpy as np

df = pd.read_csv("Company_Data.csv")
df
a=df['Sales'].mean()
for i in  range(0,400):
    if df["Sales"][i] <= a :
        df["Sales"][i]="NO"
    else:
        df["Sales"][i]="YES"

df["Sales"].value_counts()

n=11
test1 = []
test2 = []
df_cat = pd.DataFrame(test1)

df_cont = pd.DataFrame(test2)
for i in range(0,n):
    if df.iloc[:,i].dtype == "int64" or df.iloc[:,i].dtype == "float64" :
       df_cont = pd.concat([df_cont,df.iloc[:,i]],axis=1) 
    if df.iloc[:,i].dtype == "object":
       df_cat = pd.concat([df_cat,df.iloc[:,i]],axis=1)

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

df_cat
df_cat.columns
df.shape
df_cont
for i in range(0,4):
    df_cat.iloc[:,i] = LE.fit_transform(df_cat.iloc[:,i])
df_cat

df.columns
y = df_cat["Sales"]
x=pd.concat([df_cont,df_cat.iloc[:,1:]],axis=1)
x
x.columns
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=(3),train_size=0.7)

from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(max_features=4,max_samples=0.8)

RF.fit(x_train, y_train)

y_pred_train = RF.predict(x_train)
y_pred_test = RF.predict(x_test)

from sklearn.metrics import mean_squared_error

print("rmse of training :",np.sqrt(mean_squared_error(y_train, y_pred_train))) #rmse of training : 0.17896527995276465
print("rmse of testing :",np.sqrt(mean_squared_error(y_test, y_pred_test)))    #rmse of testing : 0.38800236253576253














