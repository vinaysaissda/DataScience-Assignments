# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 22:47:50 2023

@author: Vinay Sai
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder

movies = pd.read_csv("my_movies.csv")
movies

df = pd.get_dummies(movies)
df

frequent_items = apriori(df,min_support=0.1,use_colnames=True)
frequent_items1 = apriori(df,min_support=0.2,use_colnames=True)
frequent_items2 = apriori(df,min_support=0.25,use_colnames=True)
frequent_items3 = apriori(df,min_support=0.3,use_colnames=True)
frequent_items4 = apriori(df,min_support=0.05,use_colnames=True)

frequent_items
frequent_items1
frequent_items2
frequent_items3
frequent_items4

rules = association_rules(frequent_items,metric="lift",min_threshold=0.7)
rules1 = association_rules(frequent_items1,metric="lift",min_threshold=0.7)
rules2 = association_rules(frequent_items2,metric="lift",min_threshold=0.7)
rules3 = association_rules(frequent_items3,metric="lift",min_threshold=0.7)
rules4 = association_rules(frequent_items4,metric="lift",min_threshold=0.7)

rules
rules1
rules2
rules3
rules4


rules[rules.lift>1]
rules1[rules1.lift>1]
rules2[rules2.lift>1]
rules3[rules3.lift>1]
rules4[rules4.lift>1]



rules.hist()
rules1.hist()
rules2.hist()
rules3.hist()
rules4.hist()
