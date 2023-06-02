# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 13:04:35 2023

@author: Vinay Sai
"""

import pandas as pd
import numpy as np

df = pd.read_csv("Costomer+OrderForm.csv")
df
df.columns

# EDA
df.describe()

(df["Phillippines"][df["Phillippines"]=="Defective"]).count()
(df["Phillippines"][df["Phillippines"]=="Error Free"]).count()

(df["Indonesia"][df["Indonesia"]=="Defective"]).count()
(df["Indonesia"][df["Indonesia"]=="Error Free"]).count()

(df["Malta"][df["Malta"]=="Defective"]).count()
(df["Malta"][df["Malta"]=="Error Free"]).count()

(df["India"][df["India"]=="Defective"]).count()
(df["India"][df["India"]=="Error Free"]).count()

# Vizualitation

df.Phillippines.hist()
df.Indonesia.hist()
df.Malta.hist()
df.India.hist()



l = pd.DataFrame({"country":["India","Phillippines","Indonesia","Malta"],"Defective":[20/280,29/271,33/267,31/269]})
l


import researchpy as rp
table , results = rp.crosstab(l["Defective"],l["country"],test="chi-square")

table
results


# chi suare table table values for given alpha and degree of freedom
import scipy.stats as stats

crit = stats.chi2.ppf(q= 0.95, df = 9)
crit.round(4)   #  16.919


'''
                Chi-square test  results
0  Pearson Chi-square ( 9.0) =   12.0000
1                    p-value =    0.2133
2                 Cramer's V =    1.0000


Here null Hypothises is Rejected so There is a significant differenc on defective % . 


'''









