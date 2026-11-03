import numpy as np
from scipy import sparse as sp
from sklearn.neighbors import kneighbors_graph



def knn_distances(X,k):
    """
    points -> sparse D
    """

    return sp.csr_array(kneighbors_graph(X, k, mode="distance"))


def gaussian_weights(D, sigma):
    """
    sparse D -> sparse W
    """

    W = D.copy()
    W.data = np.exp(-(D.data / sigma) ** 2)    

    return W


def symmetrize(W):
    """
    includes i in j's kNN list if j in i's, symmetrizes W
    """

    return W.maximum(W.T)



def build_graph(X, k, sigma=None, metric="euclidean", p=None):
    """
    Builds sparse symmetric graph W from:
    - X (data), 
    - k (number of nearest neighbors), 
    - sigma (weight sensitivty),
    - metric (choice of distance measure)
    - p (to be implemented)
    """

    if metric == "euclidean":
        D = knn_distances(X, k)
    else: 
        raise ValueError(f"unknown metric {metric!r}") # only euclidean for now
    
    if sigma is None:
        sigma = np.median(D.data)

    return symmetrize(gaussian_weights(D, sigma))