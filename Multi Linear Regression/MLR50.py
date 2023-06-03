# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 15:57:34 2023

@author: Vinay Sai
"""

import pandas as pd
import numpy as np

df = pd.read_csv("50_Startups.csv")
df
df.boxplot(None)
df.hist()
df.dtypes

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df["State"] = LE.fit_transform(df["State"])
df["State"]

df
df.corr().to_csv("Correlation(Startups).csv")

y=df["Profit"]

x1 = "R&D_Spend"
x2 = "Administration"
x3 = "Marketing_Spend"
x4 = "State"
x5 = "R&D Spend,Administration"
x6 = "State,R&D_Spend,Administration"
x7 = "Administration,Marketing_Spend"
x8 = "Administration,State"
x9 = "State,R&D_Spend"
x10="R&D_Spend,Administration,Marketing_Spend,State"


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
LR = LinearRegression()
A = [df[["R&D_Spend"]],df[["Administration"]],df[["Marketing_Spend"]],df[["State"]],df[["R&D_Spend","Administration"]],df[["State","R&D_Spend","Administration"]],df[["Administration","Marketing_Spend"]],df[["Administration","State"]],df[["State","R&D_Spend"]],df.iloc[:,0:4]]
len(A)

B = {"column name":[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10],"rmse":[],"r2":[],"VIF":[]}
B["column name"][1]
for i in range(0,10):
    LR.fit(A[i], y)
    y_pred = LR.predict(A[i])
    B["rmse"].append(np.sqrt(mean_squared_error(y, y_pred)).round(4))
    B["r2"].append(r2_score(y, y_pred).round(4))
    B["VIF"].append(1/(1-(r2_score(y, y_pred).round(4))))
    
    # Residual Analysis
    residuals = y-y_pred
    residuals.hist()
    qqplot = sm.qqplot(residuals,line="q")
    plt.title("Normal Q-Q plot of residuals")
    plt.show()
    plt.scatter(y_pred,residuals)
    plt.show()

import statsmodels.formula.api as smf
model = smf.ols('Profit~Administration+State',data=df).fit()
model.summary()
model.resid
model.fittedvalues
model_influence = model.get_influence()
cooks,pvalue = model_influence.cooks_distance
cooks = pd.DataFrame(cooks)
cooks[0].describe    

fig = plt.subplots(figsize=(16,7))
plt.stem(np.arange(len(df)),np.round(cooks[0],3))
plt.xlabel("Row index")
plt.ylabel("Cooks Distance")
plt.show()

C = pd.DataFrame(B)
print(C)
C.to_csv("cal.csv")


"""
	column name                             	        rmse	      r2	  VIF
0	R&D Spend                   	                   9226.1005	0.9465	18.69158879
1	Administration	                                   39089.0701	0.0403	1.041992289
2	Marketing Spend                                 	26492.8294	0.5592	2.268602541
3	State	                                            39693.8072	0.0104	1.010509297
4	R&D Spend,Administration	                        9115.1979	0.9478	19.15708812
5	State,R&D Spend,Administration                     	9115.1714	0.9478	19.15708812
6	Administration,Marketing Spend	                   24927.0666	0.6097	2.562131694
7	Administration,State                            	38887.2329	0.0502	1.052853232
8	State,R&D Spend	                                   9226.1003	0.9465	18.69158879
9	R&D Spend,Administration,Marketing Spend,State	   8855.3256	0.9507	20.28397566





According to my prediction from the above table 
R&D Spend,Administration variables are getting better predictions of profit (i.e rmse=9115.1979  r2=0.9478)



    print("rmse score is : ",np.sqrt(mean_squared_error(y, y_pred)).round(4))
    print("r2 value is: ",r2_score(y, y_pred).round(4))
    
"""














