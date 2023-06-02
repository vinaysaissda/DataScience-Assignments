# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 12:39:56 2023

@author: Vinay Sai
"""

import pandas as pd
import numpy as np

df = pd.read_csv("BuyerRatio.csv")
df

df.columns
df1=df.pivot_table(columns="Observed Values")
df1
df1.Females
df1.Males

# EDA
df1.describe()

df1.boxplot(None)

# vistalizatoin

df1.Males.hist()
df1.Females.hist()

df1.plot.scatter(x="Females",y="Males")

df1.boxplot(None,vert=False)

# Hypoothises testing

x = (df1.Females.sum()/(df1.Females.sum()+df1.Males.sum()))
y = (df1.Males.sum()/(df1.Females.sum()+df1.Males.sum()))

print(x,y)

a = df1.Females.count()
b = df1.Males.count()

print(a,b)


# Two proportion test
props = np.array([x,y])
sampsize = np.array([a,b])

from statsmodels.stats.proportion import proportions_ztest
cals , pvalue = proportions_ztest(props, sampsize)

if pvalue < 0.05:
    print("H0 is Rejected")
else:
    print("H0 is Accepted")

'''
Here pvalue is 0.3785793233386694 (i.e it is greater than 0.05(Alpha))

We will Accept the Null Hypothises

Therefore there is no significant difference between male-female buyer rations are similar across regions.





'''


