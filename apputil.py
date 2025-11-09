# your code here
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple

def kmeans(X, k: int) -> Tuple[np.ndarray, np.ndarray]:
    # check input validity
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X should be 2D: (n_samples, n_features)")
    n_samples = X.shape[0]
    if not isinstance(k, (int, np.integer)) or k <= 0 or k > n_samples:
        raise ValueError("k should be a positive integer less than or equal to the number of samples")

    # scikit-learn：n_init='auto' / n_init=int
    try:
        km = KMeans(n_clusters=int(k), n_init="auto", random_state=0)
    except TypeError:
        km = KMeans(n_clusters=int(k), n_init=10, random_state=0)

    labels = km.fit_predict(X)
    centroids = km.cluster_centers_
    return centroids, labels
