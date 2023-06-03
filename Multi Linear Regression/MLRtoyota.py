# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:10:34 2023

@author: Vinay Sai
"""

import pandas as pd
import numpy as np

df= pd.read_csv("ToyotaCorolla.csv",encoding= "latin1")
df
df.columns
df.boxplot(None)
df.dtypes
df.corr()

y= df["Price"]
x =df[["Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight","Price"]]
x.corr()
x.corr().to_csv("123.csv")

x1 = "Age_08_04"
x2 = "KM"
x3 = "HP"
x4 = "cc"
x5 = "Doors"
x6 = "Gears"
x7 = "Quarterly_Tax"
x8 ="Age_08_04","HP"
x9 ="Age_08_04","cc"
x10="Age_08_04","Doors"
x11="Age_08_04","Gears"
x12="Age_08_04","Quarterly_Tax"
x13="Age_08_04","cc","HP"
x14="Age_08_04","cc","HP","Doors"
x15="Age_08_04","cc","HP","Doors","Quarterly_Tax"
x16="KM","HP","Doors"
x17="KM","Weight"
x18="KM","Weight","Gears"
x19="HP","Quarterly_Tax"
x20="HP","Gears","Quarterly_Tax"
x21 = "Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
LR = LinearRegression()

A = [df[[x1]],df[[x2]],df[[x3]],df[[x4]],df[[x5]],df[[x6]],df[["Quarterly_Tax"]],df[["Age_08_04","HP"]],df[["Age_08_04","cc"]],df[["Age_08_04","Doors"]]
     ,df[["Age_08_04","Gears"]],df[["Age_08_04","Quarterly_Tax"]],df[["Age_08_04","cc","HP"]],df[["Age_08_04","cc","HP","Doors"]],df[["Age_08_04","cc","HP","Doors","Quarterly_Tax"]]
     ,df[["KM","HP","Doors"]],df[["KM","Weight"]],df[["KM","Weight","Gears"]],df[["HP","Quarterly_Tax"]]
     ,df[["HP","Gears","Quarterly_Tax"]],df[["Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]]
len(A)


B = {"column name":[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21],"rmse":[],"r2":[],"VIF":[]}
B["column name"][18]
for i in range(0,21):
    LR.fit(A[i], y)
    y_pred = LR.predict(A[i])
    B["rmse"].append(np.sqrt(mean_squared_error(y, y_pred)).round(4))
    B["r2"].append(r2_score(y, y_pred).round(4))
    print("rmse score is : ",np.sqrt(mean_squared_error(y, y_pred)).round(4))
    print("r2 value is: ",r2_score(y, y_pred).round(4))
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
model = smf.ols('Price~Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight',data=df).fit()
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
C.to_csv("resultstoyota.csv")
"""
EXPECTED OUTPUT :
                                          column name          rmse      r2
0                                           Age_08_04       1744.8219  0.7684
1                                                  KM       2979.1386  0.3249
2                                                  HP       3441.1353  0.0992
3                                                  cc       3596.6260  0.0160
4                                               Doors       3562.8941  0.0343
5                                               Gears       3618.4753  0.0040
6                                       Quarterly_Tax       3537.5268  0.0480
7                                     (Age_08_04, HP)       1618.2941  0.8008
8                                     (Age_08_04, cc)       1738.5996  0.7701
9                                  (Age_08_04, Doors)       1733.0132  0.7715
10                                 (Age_08_04, Gears)       1731.9252  0.7718
11                         (Age_08_04, Quarterly_Tax)       1736.7722  0.7705
12                                (Age_08_04, cc, HP)       1612.7656  0.8021
13                         (Age_08_04, cc, HP, Doors)       1605.9185  0.8038
14          (Age_08_04, cc, HP, Doors, Quarterly_Tax)       1560.4821  0.8148
15                                    (KM, HP, Doors)       2886.6144  0.3661
16                                       (KM, Weight)       2162.6174  0.6442
17                                (KM, Weight, Gears)       2151.7251  0.6478
18                                (HP, Quarterly_Tax)       3228.9039  0.2069
19                         (HP, Gears, Quarterly_Tax)       3227.8219  0.2074
20  (Age_08_04,KM,HP,cc,Doors,Gears,Quarterly_Tax ,Weight)  1338.2584  0.8638
     

column name	                                                                        rmse	 r2	         VIF

Age_08_04                                                                     	1744.8219	0.7684	4.317789292
KM	                                                                           2979.1386	0.3249	1.481262035
HP                                  	                                        3441.1353	0.0992	1.110124334
cc	                                                                            3596.626	0.016	1.016260163
Doors	                                                                       3562.8941	0.0343	1.035518277
Gears	                                                                       3618.4753	0.004	1.004016064
Quarterly_Tax	                                                               3537.5268	0.048	1.050420168
('Age_08_04', 'HP')                 	                                       1618.2941	0.8008	5.020080321
('Age_08_04', 'cc')	                                                           1738.5996	0.7701	4.349717268
('Age_08_04', 'Doors')	                                                       1733.0132	0.7715	4.376367615
('Age_08_04', 'Gears')	                                                       1731.9252	0.7718	4.382120947
('Age_08_04', 'Quarterly_Tax')	                                               1736.7722	0.7705	4.357298475
('Age_08_04', 'cc', 'HP')	                                                   1612.7656	0.8021	5.0530571
('Age_08_04', 'cc', 'HP', 'Doors')	                                           1605.9185	0.8038	5.096839959
('Age_08_04', 'cc', 'HP', 'Doors', 'Quarterly_Tax')	                           1560.4821	0.8148	5.399568035
('KM', 'HP', 'Doors')	                                                       2886.6144	0.3661	1.577535889
('KM', 'Weight')                                                            	2162.6174	0.6442	2.810567735
('KM', 'Weight', 'Gears')	                                                   2151.7251	0.6478	2.839295855
('HP', 'Quarterly_Tax')                                                     	3228.9039	0.2069	1.260875047
('HP', 'Gears', 'Quarterly_Tax')	                                           3227.8219	0.2074	1.261670452
('Age_08_04', 'KM', 'HP', 'cc', 'Doors', 'Gears', 'Quarterly_Tax', 'Weight')   1338.2584	0.8638	7.342143906



     According to my prediction from the above table is:
   (Age_08_04,KM,HP,cc,Doors,Gears,Quarterly_Tax ,Weight)  
    variables are getting better predictions of profit 
    (i.e rmse = 1338.2584 and r2 = 0.8638)     
         

    """