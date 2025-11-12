# your code here
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple
import seaborn as sns
import pandas as pd
from time import perf_counter

DIAMONDS_NUMERIC: pd.DataFrame | None = None

# ex1
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

# ex2
def _load_diamonds_numeric() -> None:
    """load the numeric columns of the seaborn diamonds dataset into DIAMONDS_NUMERIC"""
    global DIAMONDS_NUMERIC
    df = sns.load_dataset("diamonds")
    cols = ["carat", "depth", "table", "price", "x", "y", "z"]
    # check columns exist
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"diamonds less: {missing}")
    DIAMONDS_NUMERIC = df[cols].copy()

# Load diamonds numeric data at module load time
if DIAMONDS_NUMERIC is None:
    _load_diamonds_numeric()

def kmeans_diamonds(n: int, k: int):
    """
    perform k-means clustering on the first n rows of the numeric columns of the seaborn diamonds dataset.
    ----
    n : int
        number of rows to use (1 <= n <= total number of rows in the dataset).
    k : int
        number of clusters.

    returns
    ----
    (centroids, labels)
      centroids: np.ndarray, shape=(k, 7)
      labels   : np.ndarray, shape=(n,)
    """
    if DIAMONDS_NUMERIC is None:
        _load_diamonds_numeric()

    total = len(DIAMONDS_NUMERIC)
    if not isinstance(n, (int, np.integer)) or n <= 0 or n > total:
        raise ValueError(f"n between 1 and {total} is required")
    # float dtype
    X = DIAMONDS_NUMERIC.head(int(n)).to_numpy(dtype=float)

    # call kmeans
    centroids, labels = kmeans(X, k)
    return centroids, labels

# ex3
def kmeans_timer(n: int, k: int, n_iter: int = 5) -> float:
    # check input validity
    if not isinstance(n_iter, (int, np.integer)) or n_iter <= 0:
        raise ValueError("n_iter should be a positive integer")

    # check DIAMONDS_NUMERIC loaded
    try:
        _ = DIAMONDS_NUMERIC  # if not defined, load it
    except NameError:
        pass
    if 'DIAMONDS_NUMERIC' in globals() and DIAMONDS_NUMERIC is None:
        _load_diamonds_numeric()

    times = []
    for _ in range(int(n_iter)):
        t0 = perf_counter()
        _ = kmeans_diamonds(n, k)   # call the function from ex2
        times.append(perf_counter() - t0)

    return float(np.mean(times))