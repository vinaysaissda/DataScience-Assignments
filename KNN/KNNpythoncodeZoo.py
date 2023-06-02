# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 11:20:14 2023

@author: Vinay Sai
"""

import pandas as pd
import numpy as np


df = pd.read_csv("Zoo.csv")
df.dtypes

df.boxplot(None)

df.columns
x_cat = df["animal name"]
x_cont = df.iloc[:,1:17]
x_cont.dtypes

from sklearn.preprocessing import StandardScaler,LabelEncoder
SS = StandardScaler()
LE = LabelEncoder()

x_cont = SS.fit_transform(x_cont)
x_cat = LE.fit_transform(x_cat)

df["animal name"].value_counts()

x_cat = pd.DataFrame(x_cat)
x_cont = pd.DataFrame(x_cont)
x= pd.concat([x_cat,x_cont],axis=1)
x
y= df["type"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=2)

# Model fitting
from sklearn.neighbors import KNeighborsRegressor
KNN = KNeighborsRegressor(n_neighbors=5,p=2)
KNN.fit(x_train,y_train)

# Predicting
y_pred_train = KNN.predict(x_train)
y_pred_test = KNN.predict(x_test)

# Metrics
from sklearn.metrics import mean_squared_error

print("rmse of training : ",np.sqrt(mean_squared_error(y_train, y_pred_train))) #rmse of training :  1.5334368495078536
print("rmse of testing : ",np.sqrt(mean_squared_error(y_test, y_pred_test)))    #rmse of testing :  1.791737307934721


"""
I got best rmse scores with n_neighbors =5 and p=2
where it is best fit according to me
"""








