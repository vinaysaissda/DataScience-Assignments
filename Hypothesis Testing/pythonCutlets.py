# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:32:09 2023

@author: Vinay Sai
"""

import pandas as pd
import numpy as np

df = pd.read_csv("Cutlets.csv")
df


# EDA
df.columns
df["Unit A"].mean()
df["Unit B"].mean()

df.describe()

df.boxplot(None)


# Visuvalaition

df.columns

df["Unit A"].hist()
df["Unit B"].hist()

df.boxplot()

'''
let
NULL HYPOTHESIS:
    H0 = mean of Unit A equals to mean of Unit B 
ALTERNATIVE HYPOTHESIS:
    H1 = mean of Unit A not equals to mean of Unit B


'''
from scipy import stats
zcal , pval = stats.ttest_ind(df["Unit A"], df["Unit B"])

if pval < 0.05:
    print("Reject Null Hypothesis")
else:
    print("Accept NUll Hypothesis")


# finding Ztable value
Ztab = stats.norm.ppf(0.95).round(4)

if zcal > Ztab:
    print("Reject Null Hypothesis")
else:
    print("Accept Null Hypothesis")

'''
CONCLUSION:
    There is no significant difference between two Units
'''






