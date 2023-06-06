# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 18:22:36 2023

@author: Vinay Sai
"""

import pandas as pd
import numpy as np

df = pd.read_csv("wine.csv")
df
df.columns

x= df.iloc[:,1:]
x
from sklearn.decomposition import PCA
pca = PCA()
pc = pca.fit_transform(x)
pc

df1 = pd.DataFrame(data=pc,columns=["pc0","pc1","pc2","pc3","pc4","pc5","pc6","pc7","pc8","pc9","pc10","pc11","pc12"])
df1

pca.explained_variance_ratio_

df_ratio = pd.DataFrame({"ratio":pca.explained_variance_ratio_,"pc":["pc0","pc1","pc2","pc3","pc4","pc5","pc6","pc7","pc8","pc9","pc10","pc11","pc12"]})
df_ratio


# pd.set_option('display.min_rows', None)
pd.set_option('display.float_format', lambda x: f'{x:.3f}') # it will convert all scientific values to normal values
df_ratio

df_new =df1.iloc[:,0:3] # Selected first three columns from pca data
df_new
df


import matplotlib.pyplot as plt
plt.scatter(df_new.iloc[:,0], df_new.iloc[:,1])
plt.show()

###############################################################################
#-- Heirarchial Clustering --##################################################
###############################################################################


import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt

plt.figure(figsize=(10,7))
plt.title("wines Dendogram")
dend = shc.dendrogram(shc.linkage(df_new,method="complete"))
dend = shc.dendrogram(shc.linkage(df_new,method="single"))
dend = shc.dendrogram(shc.linkage(df_new,method="average"))
dend = shc.dendrogram(shc.linkage(df_new,method="ward"))

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="complete")
y = cluster.fit_predict(df_new) 

y_new = pd.DataFrame(y)
y_new.value_counts()

plt.Figure(figsize=(10,7))
plt.scatter(df_new.iloc[:,0],df_new.iloc[:,1],c=cluster.labels_,cmap="rainbow")


###############################################################################
###---- K-mean---##############################################################
###############################################################################

from sklearn.cluster import KMeans
Kmean = KMeans(n_clusters=6,n_init=15)
Kmean.fit(df_new)

y1 = Kmean.predict(df_new)
y1 =pd.DataFrame(y1)
y1

y1.value_counts()

c=Kmean.cluster_centers_  
Kmean.inertia_

%matplotlib qt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df_new.iloc[:,0],df_new.iloc[:,1],df_new.iloc[:,2])
ax.scatter(c[:,0],c[:,1],c[:,2],marker="*",c="red",s=1000)
plt.show()

# Elbow method
from sklearn.cluster import KMeans
inertia =[]

for i in range(1,11):
    km = KMeans(n_clusters=i,random_state=1)
    km.fit(df_new)
    inertia.append(km.inertia_)

inertia

plt.plot(range(1,11),inertia)
plt.title("Elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.show()


###########################################################################
# Fitting for original data
###########################################################################

plt.title("wines Dendogram")
dend = shc.dendrogram(shc.linkage(x,method="complete")) # dendro gram for original data

cluster = AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="complete")
y = cluster.fit_predict(x) 

y_new = pd.DataFrame(y)
y_new.value_counts()

plt.Figure(figsize=(10,7))
plt.scatter(x.iloc[:,0],x.iloc[:,1],x.iloc[:,2],c=cluster.labels_,cmap="rainbow")

# kmean on origianl data

Kmean.fit(x)

y1 = Kmean.predict(x)
y1 =pd.DataFrame(y1)
y1

y1.value_counts()

c=Kmean.cluster_centers_  
Kmean.inertia_

%matplotlib qt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x.iloc[:,0],xiloc[:,1],x.iloc[:,2])
ax.scatter(c[:,0],c[:,1],c[:,2],marker="*",c="red",s=1000)
plt.show()

# Elbow method
from sklearn.cluster import KMeans
inertia =[]

for i in range(1,11):
    km = KMeans(n_clusters=i,random_state=1)
    km.fit(x)
    inertia.append(km.inertia_)

inertia

plt.plot(range(1,11),inertia)
plt.title("Elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.show()


"""

I found the same clusters for the data which contain first the columns of pca columns 
and same number of clusters and inertia for the original data


"""
























