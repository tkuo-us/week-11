import streamlit as st

import numpy as np
from apputil import kmeans 

X = np.array([[1, 2], [2, 3], [10, 11]])

centroids, labels = kmeans(X, k=2)

print("centroids:")
print(centroids)
print("\nlabels:")
print(labels)
