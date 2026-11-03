"""Graph construction, Laplacians, and Laplace learning."""

import numpy as np
from scipy import sparse as sp
from sklearn.neighbors import kneighbors_graph


# returns a sparse symmetric weight matrix W from data points X
def build_graph(X, k, sigma=None):
    """
    For each row (each data point), find k nearest neighbors, compute weights of their shared edges
    """
    W = sp.csr_array(kneighbors_graph(X, k, mode="distance"))
    if sigma is None:
        sigma = np.median(W.data)
    W.data = np.exp(-(W.data / sigma) ** 2)
    W = W.maximum(W.T)  # symmetrizing with OR
    return W
