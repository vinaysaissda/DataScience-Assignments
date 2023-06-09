# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 00:26:51 2023

@author: Vinay Sai
"""

import pandas as pd
import numpy as np

df= pd.read_csv("delivery_time.csv")
df
df.columns
df.shape
df.dtypes
df1 = np.log(df)
df2 = np.sqrt(df)
df3 = np.square(df)
df4 = np.exp(df)

# EDA
df.boxplot(None)
df["Delivery Time"].hist()

df["Sorting Time"].hist()

df.plot.scatter(x="Delivery Time", y="Sorting Time",title="Simple linear Regression",color="red")

df[["Sorting Time","Delivery Time"]].corr()

# spliting x and y
y = df["Delivery Time"]
x = df[["Sorting Time"]]

y1 = df1["Delivery Time"]
x1 = df1[["Sorting Time"]]

y2 = df2["Delivery Time"]
x2 = df2[["Sorting Time"]]

y3 = df3["Delivery Time"]
x3 = df3[["Sorting Time"]]

y4 = df4["Delivery Time"]
x4 = df4[["Sorting Time"]]


# Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR1 = LinearRegression()
LR2 = LinearRegression()
LR3 = LinearRegression()
LR4 = LinearRegression()


LR.fit(x,y)
LR1.fit(x1,y1)
LR2.fit(x2,y2)
LR3.fit(x3,y3)
LR4.fit(x4,y4)


LR.intercept_

# predicting
y_pred = LR.predict(x)
y_pred1 = LR.predict(x1)
y_pred2 = LR.predict(x2)
y_pred3 = LR.predict(x3)
y_pred4 = LR.predict(x4)


import matplotlib.pyplot as plt
plt.scatter(x= x, y=y)
plt.plot(x, y_pred,color = "red")
plt.show()

# Metrics
from sklearn.metrics import mean_squared_error

print("Root mean square error is : ",np.sqrt(mean_squared_error(y, y_pred))) 
print("Root mean square error over log is : ",np.sqrt(mean_squared_error(y1, y_pred1))) 
print("Root mean square error over square root is : ",np.sqrt(mean_squared_error(y2, y_pred2))) 
print("Root mean square error over square  is : ",np.sqrt(mean_squared_error(y3, y_pred3))) 
print("Root mean square error over exponential is : ",np.sqrt(mean_squared_error(y4, y_pred4))) 

'''
EXPECTED OUTPUT :
    Root mean square error is :  2.7916503270617654
    Root mean square error over log is :  6.677169733001051
    Root mean square error over square root is :  6.5592973174198095
    Root mean square error over square  is :  267.1933493745269
    Root mean square error over exponential is :  857907131835.6295
'''














