import numpy as np

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
