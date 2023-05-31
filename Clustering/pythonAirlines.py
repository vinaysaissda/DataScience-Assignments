# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 19:02:04 2023

@author: Vinay Sai
"""

import pandas as pd
import numpy as np

df = pd.read_csv("EastWestAirlines.csv")
df

df.boxplot(None)
df.dtypes


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

cluster = AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")
y = cluster.fit_predict(df)

y = pd.DataFrame(y)
y.value_counts()


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
dbscan = DBSCAN(eps=2,min_samples=3)
dbscan.fit(df)

dbscan.labels_
c1 = pd.DataFrame(dbscan.labels_,columns=["Cluster"])
c1

c1["Cluster"].value_counts()





































