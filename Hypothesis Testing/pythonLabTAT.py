# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:55:56 2023

@author: Vinay Sai
"""

import pandas as pd
import numpy as np

df = pd.read_csv("LabTAT.csv")
df

# EDA
df.describe()
df.boxplot(None)

# Visualition

df.columns

df["Laboratory 1"].hist()
df["Laboratory 2"].hist()
df["Laboratory 3"].hist()
df["Laboratory 4"].hist()

df.boxplot(vert=False)
df

# Model fitting
import scipy.stats as stats
fval , pval = stats.f_oneway(df["Laboratory 1"],df["Laboratory 2"],df["Laboratory 3"],df["Laboratory 4"])
print(fval,pval)

if pval < 0.05:
    print("Reject Null Hypothesis")
else:
    print("Accept Null Hypothesis")
    
"""
LET :
    NULL HYPOTHESIS H0 :
        let mean of Laboratory 1 is equals to mean of Laboratory 2 is equals to mean of Laboratory 3 is equals to mean of Laboratory 4
    ALTERNATIVE HYPOTHESIS H1 :
        Not all means are equal

CONCLUSION :
    As H0 is rejected so not all means are equal


"""