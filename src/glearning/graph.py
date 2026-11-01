"""Graph construction, Laplacians, and Laplace learning."""

import numpy as np
from scipy import sparse as sp
from sklearn.neighbors import kneighbors_graph


###
### Graph construction
###

# returns a sparse symmetric weight matrix W from data points X and possibly a choice of distance
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


###
### Laplacians and energy
###

# Takes a dense symmetric weight matrix and returns the graph laplacian L=D-W
# D is the diagonal matrix of degrees
def laplacian(W):
    return np.diag(W.sum(axis=1)) - W


def laplacian_sparse(W):
    W_array = sp.csr_array(W)
    return sp.diags_array(W_array.sum(axis=1)) - W_array


# Direct computation of the dirichlet energy
def dirichlet_energy(W, u):
    return 0.5 * (W * (np.subtract.outer(u, u)) ** 2).sum(axis=1).sum(axis=0)


###
### Laplace learning
###

# Takes the graph laplacian, a 1D array of the indices with labels, and a 1D array of labels corresponding to said indices
# returns the full vector solution u
def laplace_learning(L, labeled, y):
    n = L.shape[0]
    unlabeled = np.setdiff1d(np.arange(n), labeled)  # creates unlabeled indices
    Luu = L[np.ix_(unlabeled, unlabeled)]
    Lul = L[np.ix_(unlabeled, labeled)]
    b = -Lul @ y
    uu = np.linalg.solve(Luu, b)
    u = np.zeros(n)
    u[labeled] = y
    u[unlabeled] = uu
    return u


def slaplace_learning():
    """
    version of laplace_learning that takes advantage of sparse arrays
    """
    raise NotImplementedError


###
### Agreement
###

def kagreement(*args):
    return sum(p**2 for p in args) + (1 - sum(p for p in args)) ** 2
