# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 02:48:19 2023

@author: Vinay Sai
"""

import pandas as pd
import numpy as np

df= pd.read_csv("Salary_Data.csv")
df

df.dtypes
df.shape
df.columns
df1 = np.log(df)
df2 = np.sqrt(df)
df3 = np.square(df)

# EDA
df.boxplot(None)
df["YearsExperience"].hist()
df["Salary"].hist()

df.plot.scatter(x="YearsExperience", y="Salary",title="Simple linear Regression",color="red")

df[["YearsExperience","Salary"]].corr()

# spliting x and y
y = df["YearsExperience"]
x = df[["Salary"]]

y1 = df1["YearsExperience"]
x1 = df1[["Salary"]]

y2 = df2["YearsExperience"]
x2 = df2[["Salary"]]

y3 = df3["YearsExperience"]
x3 = df3[["Salary"]]



# Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x,y)

LR1 = LinearRegression()
LR1.fit(x1,y1)

LR2 = LinearRegression()
LR2.fit(x2,y2)

LR3 = LinearRegression()
LR.fit(x3,y3)


LR.intercept_

# predicting
y_pred = LR.predict(x)
y_pred1 = LR.predict(x1)
y_pred2 = LR.predict(x2)
y_pred3 = LR.predict(x3)


import matplotlib.pyplot as plt
plt.scatter(x= x, y=y)
plt.plot(x, y_pred,color = "red")
plt.show()

# Metrics
from sklearn.metrics import mean_squared_error

print("Root mean square error is : ",np.sqrt(mean_squared_error(y, y_pred))) 
print("Root mean square error over log is : ",np.sqrt(mean_squared_error(y1, y_pred1))) 
print("Root mean square error over square root is : ",np.sqrt(mean_squared_error(y2, y_pred2))) 
print("Root mean square error over square is : ",np.sqrt(mean_squared_error(y3, y_pred3))) 

'''
EXPECTED OUTPUT:
    Root mean square error is :  17.71229230589682
    Root mean square error over log is :  13.697141597177307
    Root mean square error over square root is :  14.4102835843467
    Root mean square error over square is :  7.117486572354788
'''