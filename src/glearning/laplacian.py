import numpy as np
from scipy import sparse as sp

# Takes a dense symmetric weight matrix and returns the graph laplacian L=D-W
# D is the diagonal matrix of degrees
def laplacian(W):
    return np.diag(W.sum(axis=1)) - W

# Sparse version of laplacian()
def laplacian_sparse(W):
    W_array = sp.csr_array(W)
    return sp.diags_array(W_array.sum(axis=1)) - W_array

# Direct computation of the dirichlet energy
def dirichlet_energy(W, u):
    return 0.5 * (W * (np.subtract.outer(u, u)) ** 2).sum(axis=1).sum(axis=0)