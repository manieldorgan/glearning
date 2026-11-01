"""Float comparisons and sanity checks on weight matrices."""

import numpy as np
from scipy import sparse as sp


###
### Float comparisons
###

def float_eq(float1, float2, tolerance=1e-12) -> bool:
    return abs(float1 - float2) < abs(tolerance)


# checks if float1 is greater than or equal to float2
def float_geq(float1, float2, tolerance=1e-12) -> bool:
    return float1 >= float2 - tolerance


# checks if float1 is less than or equal to float2
def float_leq(float1, float2, tolerance=1e-12) -> bool:
    return float1 <= float2 + tolerance


###
### Weight matrix checks
###

def check_min_neighbors(W, k) -> bool:
    W = sp.csr_array(W)
    W.eliminate_zeros()
    counts = np.diff(W.indptr)  # number of stored entries in each row
    return bool(np.all(counts >= k))


def check_diag_sum(W) -> bool:
    return float_eq(W.diagonal().sum(axis=0), 0.0)


def check_symmetric(W) -> bool:
    return float_eq(np.abs(W - W.T).max(), 0.0)
