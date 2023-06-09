# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 19:34:41 2023

@author: Vinay Sai
"""

import pandas as pd
import numpy as np
 
df = pd.read_csv("book.csv",encoding="latin1")

df

df = df.iloc[:,1:]
df

df.sort_values("User.ID")
df["Book.Rating"].value_counts()

df["Book.Rating"].hist()
len(df["Book.Rating"].unique())

df["Book.Rating"].value_unique()

df["User.ID"].fillna(0,inplace=True)
df.columns

user_df = df.pivot_table(index="User.ID",columns="Book.Title",values = "Book.Rating")
user_df

user_df.iloc[0]
user_df.iloc[200]
len(list(user_df))

user_df.fillna(0,inplace=True)
user_df


from scipy.spatial.distance import cosine,correlation
from sklearn.metrics import pairwise_distances
user_sim = 1 - pairwise_distances(user_df.values,metric="cosine")
user_sim.shape

# store the results in dataframe
user_sim_df = pd.DataFrame(user_sim)
user_sim_df

# seting the index and column names to user ids
user_sim_df.index = df["User.ID"].unique()
user_sim_df.columns = df["User.ID"].unique()

user_sim_df
user_sim_df.iloc[0:5,0:5]

np.fill_diagonal(user_sim, 0)

user_sim_df.to_csv("cosin_cal.csv")


# Most similar Users
user_sim_df.max()
user_sim_df.idxmax(axis=1).to_csv("recommending_table.csv")

 