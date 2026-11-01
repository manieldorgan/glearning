import numpy as np
from sklearn.neighbors import kneighbors_graph 
from scipy import sparse as sp






###
### BASIC
###

def float_eq(float1, float2, tolerance=1e-12)->bool:
    return True if (abs(float1-float2) < abs(tolerance)) else False

# checks if float1 is greater than or equal to float2
def float_geq(float1, float2, tolerance=1e-12):
    return True if (float1 >= float2 - tolerance) else False

# checks if float1 is less than or equal to float2
def float_leq(float1, float2, tolerance=1e-12):
    return True if (float1 <= float2 + tolerance) else False

# Creates a random symmetric numpy array of size n
# sample drawn from the standard normal distribition
def randsym(n):
    A = np.random.standard_normal((n,n))
    return (A + A.T) / 2

###
### Experiment Checking
###

def check_min_neighbors(W, k):
    W = sp.csr_array(W)
    W.eliminate_zeros()              
    counts = np.diff(W.indptr)          
    return np.all(counts >= k)

def check_diag_sum(W):
    return float_eq(W.diagonal().sum(axis=0), 0.0)

def check_symmetric(W):
    return float_eq(np.abs(W-W.T).max(), 0.0)


###
### Graph Learning
###  

def kagreement(*args):
    return sum(p**2 for p in args)+(1 - sum(p for p in args))**2

# Takes a dense symmetric weight matrix and returns the graph laplacian L=D-W
# D is the diagonal matrix of degrees
def laplacian(W):
    return np.diag(W.sum(axis=1)) - W

# Direct computation of the dirichlet energy
def dirichlet_energy(W, u):
    return 0.5*(W*(np.subtract.outer(u,u))**2).sum(axis=1).sum(axis=0)

# Takes the graph laplacian, a 1D array of the indices with labels, and a 1D array of labels corresponding to said indices
# returns the full vector solution u
def laplace_learning(L, labeled, y):
    n = L.shape[0]
    unlabeled = np.setdiff1d(np.arange(n), labeled) # creates unlabeled indices
    Luu = L[np.ix_(unlabeled, unlabeled)]
    Lul = L[np.ix_(unlabeled, labeled)]
    b = -Lul@y
    uu = np.linalg.solve(Luu, b)
    u = np.zeros(n)
    u[labeled] = y; u[unlabeled] = uu
    return u

# returns a sparse symmetric weight matrix W from data points X and possibly a choice of distance
def build_graph(X, k, sigma=None):
    """
    For each row (each data point), find k nearest neighbors, compute weights of their shared edges
    """
    W = sp.csr_array(kneighbors_graph(X, k, mode='distance'))
    if sigma is None: sigma = np.median(W.data)
    W.data = np.exp(-(W.data/sigma)**2)
    W = W.maximum(W.T)    # symmetrizing with OR
    return W
    
def laplacian_sparse(W):
    W_array = sp.csr_array(W)
    return sp.diags_array(W_array.sum(axis=1)) - W_array

def slaplace_learning():
    """
    version of laplace_learning that takes advantage of sparse arrays
    """
    ...


if __name__ == "__main__":
    # Checks for laplacian
    W = np.array([[0,1,0,0], [1,0,1,0], [0,1,0,1], [0,0,1,0]])
    print(laplacian(W))
    print(True if np.all(laplacian(W)@np.ones(4, ) == 0) else False)
    print(laplace_learning(laplacian(W), np.array([0]), np.array([1])))

    # Checks for dirichlet energy
    u1 = [1, 0.8, -0.7, -1]
    print(u1@laplacian(W)@u1)
    print(True if float_eq(dirichlet_energy(W, u1), 2.38) else False)
    print(True if float_eq(dirichlet_energy(W, u1), u1@laplacian(W)@u1) else False)
    
    A = randsym(4)
    u2 = np.random.standard_normal((4,))
    print(True if float_eq(dirichlet_energy(A, u2), u2@laplacian(A)@u2) else False)

    # Checks for laplace_learning
    w2 = np.array([[0, 1, 1, 0, 0, 0], 
                   [1, 0, 1, 0, 0, 0], 
                   [1, 1, 0, 0.1, 0, 0],
                   [0, 0, 0.1, 0, 1, 1],
                   [0, 0, 0, 1, 0, 1],
                   [0, 0, 0, 1, 1, 0]
                   ])
    d = np.diag(w2.sum(axis=1))
    L = d - w2
    labeled = np.array([0, 5])
    labels = np.array([1, -1])
    print(laplace_learning(L, labeled, labels))

    w3 = np.array([[0, 1, 1, 0, 0, 0], 
                   [1, 0, 1, 0, 0, 0], 
                   [1, 1, 0, 1, 0, 0], # connection between {1, 2, 3} triangle and {4, 5, 6} triangle
                   [0, 0, 1, 0, 1, 1],
                   [0, 0, 0, 1, 0, 1],
                   [0, 0, 0, 1, 1, 0]
                   ])
    d2 = np.diag(w3.sum(axis=1))
    L = d2 - w3
    labeled2 = np.array([0, 5])
    labels2 = np.array([1, -1])
    print(laplace_learning(L, labeled2, labels2))


    # Checks for build_graph
    """
    need to:
    compute all pairwise distances with scipy.spatial.distance.cdist(X, X) 
    find each point’s k nearest with np.argsort, 
    confirm graph has those edges.

    confirm there are at least k nonzero entries per row
    """

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    A = build_graph(X, 3)
    print(A.toarray())
    if (float_eq(A.diagonal().sum(axis=0), 0.0) & float_eq(np.abs(A-A.T).max(), 0.0)):
        print("good build")
    else:
        print("bad build")

    # Checks for laplacian sparse
    W_path = np.array([[0, 1, 0, 0],
                       [1, 0, 1, 0],
                       [0, 1, 0, 1],
                       [0, 0, 1, 0]], dtype=float)
    L_expected = np.array([[ 1, -1,  0,  0],
                           [-1,  2, -1,  0],
                           [ 0, -1,  2, -1],
                           [ 0,  0, -1,  1]], dtype=float)
    L = laplacian_sparse(W_path)
    assert sp.issparse(L), "result should be sparse"
    assert np.allclose(L.toarray(), L_expected), "path graph Laplacian is wrong"


"""
def check_min_neighbors(W, k):
    W = sp.csr_array(W)
    W.eliminate_zeros()                 # drop any entries that are exactly 0.0
    counts = np.diff(W.indptr)          # number of stored entries in each row
    assert np.all(counts >= k), (
        f"some rows have fewer than k={k} neighbors; min is {counts.min()} "
        f"(rows: {np.where(counts < k)[0][:10]})"
    )
    print(f"neighbor check passed: min {counts.min()}, max {counts.max()}, mean {counts.mean():.1f}")
"""