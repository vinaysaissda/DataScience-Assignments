# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 01:27:39 2023

@author: Vinay Sai
"""

import pandas as pd
import numpy as np

df = pd.read_csv("Airlines+Data.csv")
df

df.plot()

from matplotlib import pyplot
df = pd.read_csv("Airlines+Data.csv",header=0,index_col=0,parse_dates=True)
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

df.Month
df["Date"] = pd.to_datetime(df.Month,format="%b-%y")
df

df["month"] = df.Date.dt.strftime("%b")
df["year"] = df.Date.dt.strftime("%Y")
df

# HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,8))
heatmap_y_month = pd.pivot_table(data=df,values="Passengers",index='year',columns='month',fill_value=(0))
sns.heatmap(heatmap_y_month,annot=True,fmt="g")

# Boxplot
plt.Figure(figsize=(10,6))
sns.boxplot(x='month',y='Passengers',data=df)


# Line plot
plt.Figure(figsize=(10,6))
sns.lineplot(x='month',y='Passengers',data=df)


t = []
for i in range(1,97):
           t.append(i)  

t = pd.DataFrame(data=t,columns=['t'])

df = pd.concat([df,t],axis=1)

t_square = []
for i in range(1,97):
           t_square.append(i*i)  

t_square = pd.DataFrame(data=t_square,columns=['t_square'])

df = pd.concat([df,t_square],axis=1)

log_Passengers = []
for i in range(0,96):
    log_Passengers.append(np.log(df['Passengers'][i]))

log_Passengers = pd.DataFrame(data=log_Passengers,columns=['log_Passengers'])
df = pd.concat([df,log_Passengers],axis=1)

df2 = df[["month"]]

from sklearn.preprocessing import OneHotEncoder
OE = OneHotEncoder()
df3 = OE.fit_transform(df2).toarray()

df4 = pd.DataFrame(data=df3,columns=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
df4

df = pd.concat([df,df4],axis=1)

# spliting the data
df.shape
train = df.head(76)
test = df.tail(20)


# Linear model
import statsmodels.formula.api as smf
linear_model = smf.ols('Passengers~t',data=train).fit()
pred_linear = pd.Series(linear_model.predict(pd.DataFrame(test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(pred_linear))**2))

rmse_linear  # 57.00014788256592

# Exponentail
exp = smf.ols('log_Passengers~t',data=train).fit()
pred_exp = pd.Series(exp.predict(pd.DataFrame(test['t'])))
rmse_exp = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(pred_exp)))**2))

rmse_exp  # 46.62154394719422

# Quadratic
quad = smf.ols('Passengers~t+t_square',data=train).fit()
pred_quad = pd.Series(quad.predict(test[['t','t_square']]))
rmse_quad = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(pred_quad))**2))

rmse_quad  # 58.494274639472664


# Additive seasonality

add_sea = smf.ols('Passengers~jan+feb+mar+apr+may+jun+jul+aug+sep+oct+nov+dec',data=train).fit()
pred_add_sea = pd.Series(add_sea.predict(test[['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']]))
rmse_add_sea = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(pred_add_sea))**2))

rmse_add_sea #  132.25413439949475


# Additive Seasonality Quadratic

add_sea_quad = smf.ols('Passengers~t+t_square+jan+feb+mar+apr+may+jun+jul+aug+sep+oct+nov+dec',data=train).fit()
pred_add_sea_quad = pd.Series(add_sea_quad.predict(test[['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(pred_add_sea_quad))**2))

rmse_add_sea_quad  # 39.759766338063756


# Multiplicative Seasonality

mul_sea = smf.ols('log_Passengers~jan+feb+mar+apr+may+jun+jul+aug+sep+oct+nov+dec',data=train).fit()
pred_mult_sea = pd.Series(mul_sea.predict(test))
rmse_mult_sea = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(pred_mult_sea)))**2))

rmse_mult_sea   #  137.6109085625647


# Multiplicative Additive Seasonality

mul_add_sea = smf.ols('log_Passengers~t+jan+feb+mar+apr+may+jun+jul+aug+sep+oct+nov+dec',data=train).fit()
pred_mult_add_sea = pd.Series(mul_add_sea.predict(test))
rmse_mult_add_sea = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(pred_mult_add_sea)))**2))

rmse_mult_add_sea   #  11.784250178871144



# Compare results
data = {"Model":pd.Series(["linear_model","exp",'quad',"add_sea","add_sea_quad","mul_sea","mul_add_sea"]),"RMSE_values":pd.Series([rmse_linear,rmse_exp,rmse_quad,rmse_add_sea,rmse_add_sea_quad,rmse_mult_sea,rmse_mult_add_sea])}
type(data)

table_rmse = pd.DataFrame(data)
table_rmse.sort_values(["RMSE_values"])


'''
Expected output :
          Model  RMSE_values
0  linear_model    57.000148
1           exp    46.621544
2          quad    58.494275
3       add_sea   132.254134
4  add_sea_quad    39.759766
5       mul_sea   137.610909
6  mul_add_sea    11.784250

Here by using "Multiplicative Additive Seasonality" MODEL we are geting less rmse score

'''




