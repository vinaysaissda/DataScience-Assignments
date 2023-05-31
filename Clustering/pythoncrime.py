# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 11:43:16 2023

@author: Vinay Sai
"""
import pandas as pd
import numpy as np

df = pd.read_csv("crime_data.csv")
df

df.boxplot(None)
df.dtypes

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df["Unnamed: 0"] = LE.fit_transform(df["Unnamed: 0"])

# Hirarical clustering

import scipy.cluster.hierarchy as hc
import matplotlib.pyplot as plt

plt.figure(figsize=(10,7))
plt.title("Crime Dendrogram")
hc.dendrogram(hc.linkage(df,method="complete"))
hc.dendrogram(hc.linkage(df,method="single"))
hc.dendrogram(hc.linkage(df,method="average"))
hc.dendrogram(hc.linkage(df,method="ward"))


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="complete")
y = cluster.fit_predict(df)

y = pd.DataFrame(y)
y.value_counts()
df

###########################################################################
# Kmean
###########################################################################

from sklearn.cluster import KMeans
Kmean = KMeans(n_clusters=6,n_init=15)

y= Kmean.fit_predict(df)
y = pd.DataFrame(y)
y.value_counts()

Kmean.inertia_
c = Kmean.cluster_centers_
c


inertia = []

for i in range(1,11):
    Kmean = KMeans(n_clusters=i,random_state=(1))
    Kmean.fit(df)
    inertia.append(Kmean.inertia_)

inertia

plt.plot(range(1,11),inertia)
plt.title("Elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.show()

###############################################################################
# DBscan
###############################################################################

df
df.info()

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=1,min_samples=2)
dbscan.fit(df)

dbscan.labels_
c1 = pd.DataFrame(dbscan.labels_,columns=["Cluster"])
c1

c1["Cluster"].value_counts()
