import streamlit as st

import numpy as np
from apputil import kmeans, kmeans_diamonds 

X = np.array([[1, 2], [2, 3], [10, 11]])
centroids, labels = kmeans(X, k=2)
print("centroids:")
print(centroids)
print("\nlabels:")
print(labels)

centroids, labels = kmeans_diamonds(n=1000, k=5)
print("\nex2:")
print(centroids.shape)  # (5, 7)
print(labels.shape)     # (1000,)