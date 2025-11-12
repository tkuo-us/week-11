import streamlit as st
import numpy as np
from apputil import kmeans, kmeans_diamonds, kmeans_timer

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

# ex3
# fix n=100,5000,...,45000，k=5
n_values = np.arange(100, 50000, 5000)
k5_times = [kmeans_timer(n, 5, n_iter=5) for n in n_values]

print("n average seconds:")
for n, t in zip(n_values, k5_times):
    print(f"n={n:<6} -> average time {t:.4f} 秒")

# fix n=10000, changing k=2,...,9
k_values = np.arange(2, 10)
n10k_times = [kmeans_timer(10000, k, n_iter=3) for k in k_values]

print("\n k average time(seconds):")
for k, t in zip(k_values, n10k_times):
    print(f"k={k:<3} -> average time {t:.4f} sec")