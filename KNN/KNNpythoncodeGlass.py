# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 10:45:02 2023

@author: Vinay Sai
"""

import pandas as pd
import numpy as np


df = pd.read_csv("glass.csv")
df

df.dtypes

y= df["Type"]
x= df.iloc[:,:9]
x


# StandardScaler
from  sklearn.preprocessing import StandardScaler
SS = StandardScaler()
x_ss = SS.fit_transform(x)
x_ss

# Data partition
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_ss,y,train_size=0.8,random_state=2)

# Model fitting
from sklearn.neighbors import KNeighborsRegressor
KNN = KNeighborsRegressor(n_neighbors=3,p=2)
KNN.fit(x_train,y_train)

# prediction
y_pred_train = KNN.predict(x_train)
y_pred_test = KNN.predict(x_test)

# metrics
from sklearn.metrics import mean_squared_error

print("rmse of training : ",np.sqrt(mean_squared_error(y_train, y_pred_train))) #rmse of training :  0.7761051354594619
print("rmse of testing : ",np.sqrt(mean_squared_error(y_test, y_pred_test)))    #rmse of testing :  0.9671610604216165

"""
I got best rmse scores with n_neighbors =3 and p=2

"""



