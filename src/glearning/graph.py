import numpy as np
from scipy import sparse as sp
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

### Helper
def symmetrize(W):
    """
    includes i in j's kNN list if j in i's, symmetrizes W
    """

    return W.maximum(W.T)


### Stage 1
def knn_distances(X,k):
    """
    points, num of nearest neighbors -> sparse D
    """
    return sp.csr_array(kneighbors_graph(X, k, mode="distance"))

def epsilon_distances(X, eps):
    """
    Points, eps -> sparse D 
    """
    # handle non unique points by giving them small float weights that wont effect calculations 
    return sp.csr_array(radius_neighbors_graph(X, eps, mode="distance"))



### Stage 2
def gaussian_weights(D, sigma):
    """
    sparse D -> sparse W
    """

    W = D.copy()
    W.data = np.exp(-(D.data / sigma) ** 2)    

    return W


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