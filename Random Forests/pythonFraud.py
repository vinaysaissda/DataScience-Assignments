# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 13:08:34 2023

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

from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(max_depth=(4))
RF.fit(x_train, y_train)

y_pred_train = RF.predict(x_train)
y_pred_test = RF.predict(x_test)


from sklearn.metrics import mean_squared_error

print("rmse of training : ",np.sqrt(mean_squared_error(y_train, y_pred_train))) #rmse of training :  0.3673151320080401
print("rmse of testing : ",np.sqrt(mean_squared_error(y_test, y_pred_test)))    #rmse of testing :  0.4100854075052785



