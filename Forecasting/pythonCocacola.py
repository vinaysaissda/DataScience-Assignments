# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 20:54:05 2023

@author: Vinay Sai
"""

import pandas as pd
import numpy as np

df = pd.read_csv("CocaCola_Sales_Rawdata.csv")
df
df.columns

df.plot()

from matplotlib import pyplot
df = pd.read_csv("CocaCola_Sales_Rawdata.csv",header=0,index_col=0,parse_dates=True)
df.plot(kind="kde")
pyplot.show()

# Lag plot
from pandas.plotting import lag_plot
lag_plot(df)
pyplot.show()

# autocorrelation plot 
pyplot.Figure(figsize=(40,10))
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df, lags=90)

df.columns

df['quarter'] = 0 # quater column created
for i in range(42):
    p=df['Quarter'][i]
    df['quarter'][i]=p[0:2]

df["year"] = 0
for i in range(42):
    d = df["Quarter"][i]
    df["year"][i] = int(d[3:5])

df
    
df_dummies=pd.DataFrame(pd.get_dummies(df['quarter']),columns=['Q1','Q2','Q3','Q4'])
df=pd.concat([df,df_dummies],axis= 1)    
df    

# HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,8))
heatmap_y_month = pd.pivot_table(data=df,values="Sales",index='year',columns='quarter',fill_value=(0))
sns.heatmap(heatmap_y_month,annot=True,fmt="g")


# Boxplot
plt.Figure(figsize=(10,6))
sns.boxplot(x='quarter',y='Sales',data=df)


df

t = []
for i in range(1,43):
           t.append(i)  

t = pd.DataFrame(data=t,columns=['t'])

df = pd.concat([df,t],axis=1)

t_square = []
for i in range(1,43):
           t_square.append(i*i)  

t_square = pd.DataFrame(data=t_square,columns=['t_square'])

df = pd.concat([df,t_square],axis=1)

df.columns
log_Sales = []
for i in range(0,42):
    log_Sales.append(np.log(df['Sales'][i]))

log_Sales = pd.DataFrame(data=log_Sales,columns=['log_Sales'])
df = pd.concat([df,log_Sales],axis=1)

# spliting the data
df.shape
train = df.head(30)
test = df.tail(12)

train
test



# Linear model
import statsmodels.formula.api as smf
linear_model = smf.ols('Sales~t',data=train).fit()
pred_linear = pd.Series(linear_model.predict(pd.DataFrame(test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_linear))**2))

rmse_linear  # 714.0144483818335

# Exponentail
exp = smf.ols('log_Sales~t',data=train).fit()
pred_exp = pd.Series(exp.predict(pd.DataFrame(test['t'])))
rmse_exp = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_exp)))**2))

rmse_exp  # 552.2821039688208

# Quadratic
quad = smf.ols('Sales~t+t_square',data=train).fit()
pred_quad = pd.Series(quad.predict(test[['t','t_square']]))
rmse_quad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_quad))**2))

rmse_quad  # 646.2715428655371


# Additive seasonality

add_sea = smf.ols('Sales~Q1+Q2+Q3+Q4',data=train).fit()
pred_add_sea = pd.Series(add_sea.predict(test[["Q1","Q2","Q3",'Q4']]))
rmse_add_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_add_sea))**2))

rmse_add_sea #  1778.0065467723998


# Additive Seasonality Quadratic

add_sea_quad = smf.ols('Sales~t+t_square+Q1+Q2+Q3+Q4',data=train).fit()
pred_add_sea_quad = pd.Series(add_sea_quad.predict(test[["Q1","Q2","Q3",'Q4','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_add_sea_quad))**2))

rmse_add_sea_quad  # 586.0533068427089


# Multiplicative Seasonality

mul_sea = smf.ols('log_Sales~Q1+Q2+Q3+Q4',data=train).fit()
pred_mult_sea = pd.Series(mul_sea.predict(test[["Q1","Q2","Q3",'Q4']]))
rmse_mult_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_mult_sea)))**2))

rmse_mult_sea   #  1828.923891189183


# Multiplicative Additive Seasonality

mul_add_sea = smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data=train).fit()
pred_mult_add_sea = pd.Series(mul_add_sea.predict(test[["Q1","Q2","Q3",'Q4','t']]))
rmse_mult_add_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_mult_add_sea)))**2))

rmse_mult_add_sea   # 410.2497060538072


# Compare results
data = {"Model":pd.Series(["linear_model","exp",'quad',"add_sea","add_sea_quad","mul_sea","mul_add_sea"]),"RMSE_values":pd.Series([rmse_linear,rmse_exp,rmse_quad,rmse_add_sea,rmse_add_sea_quad,rmse_mult_sea,rmse_mult_add_sea])}
type(data)

table_rmse = pd.DataFrame(data)
table_rmse.sort_values(["RMSE_values"])


"""
          Model  RMSE_values
6   mul_add_sea   410.249706
1           exp   552.282104
4  add_sea_quad   586.053307
2          quad   646.271543
0  linear_model   714.014448
3       add_sea  1778.006547
5       mul_sea  1828.923891

Here by using "Multiplicative Additive Seasonality" MODEL we are geting less rmse score


"""











