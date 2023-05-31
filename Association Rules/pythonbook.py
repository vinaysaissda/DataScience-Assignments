# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:16:44 2023

@author: Vinay Sai
"""

import pandas as pd
import numpy as np

df  = pd.read_csv("book.csv")
df

trans = []

for i in range(0,2000):
    trans.append([str(df.values[i,j]) for j in range(0,11)])
        
trans

from apyori import apriori
rules = apriori(transactions=trans,min_support=0.001,min_confidence=0.001,min_lift=1,min_length=1,max_length=None)
rules1 = apriori(transactions=trans,min_support=0.01,min_confidence=0.001,min_lift=1,min_length=2,max_length=None)
rules2 = apriori(transactions=trans,min_support=0.001,min_confidence=0.01,min_lift=1,min_length=3,max_length=None)
rules3 = apriori(transactions=trans,min_support=0.02,min_confidence=0.05,min_lift=1,min_length=4,max_length=None)
rules4 = apriori(transactions=trans,min_support=0.05,min_confidence=0.1,min_lift=1,min_length=2,max_length=None)

results = list(rules)
results1 = list(rules1)
results2 = list(rules2)
results3 = list(rules3)
results4 = list(rules4)


results
results1
results2
results3
results4

len(results)
len(results1)
len(results2)
len(results3)
len(results4)




def inspect(results):
    lhs = [tuple(results[2][0])for results in results]
    rhs = [tuple(results[2][0][1][0])for results in results]
    support = [results[1]for results in results]
    confidence = [results[2][0][2]for results in results]
    lifts = [results[2][0][3]for results in results]
    return list(zip(lhs,rhs,support,confidence,lifts))

resultsinDataframe = pd.DataFrame(inspect(results),columns=["left hand side","right hand side","support","Confidence","Lift"])
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    






