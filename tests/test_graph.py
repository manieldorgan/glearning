"""Tests converted from the checks that used to live in reuse.py's __main__ block.

Run from the repo root with:  pytest
"""

import numpy as np
from scipy import sparse as sp

from glearning import (
    build_graph,
    check_diag_sum,
    check_min_neighbors,
    check_symmetric,
    dirichlet_energy,
    float_eq,
    laplace_learning,
    laplacian,
    laplacian_sparse,
)

# 4-node path graph: 0 - 1 - 2 - 3
W_PATH = np.array([[0, 1, 0, 0],
                   [1, 0, 1, 0],
                   [0, 1, 0, 1],
                   [0, 0, 1, 0]], dtype=float)

L_PATH = np.array([[ 1, -1,  0,  0],
                   [-1,  2, -1,  0],
                   [ 0, -1,  2, -1],
                   [ 0,  0, -1,  1]], dtype=float)


# Creates a random symmetric numpy array of size n
# sample drawn from the standard normal distribution
def randsym(n, rng):
    A = rng.standard_normal((n, n))
    return (A + A.T) / 2


###
### Laplacian
###

def test_laplacian_path():
    assert np.allclose(laplacian(W_PATH), L_PATH)


def test_laplacian_kills_constants():
    assert np.allclose(laplacian(W_PATH) @ np.ones(4), 0)


def test_laplacian_sparse_matches_dense():
    L = laplacian_sparse(W_PATH)
    assert sp.issparse(L), "result should be sparse"
    assert np.allclose(L.toarray(), L_PATH), "path graph Laplacian is wrong"


###
### Dirichlet energy
###

def test_dirichlet_energy_known_value():
    u = np.array([1, 0.8, -0.7, -1])
    assert float_eq(dirichlet_energy(W_PATH, u), 2.38)


def test_dirichlet_energy_equals_quadratic_form():
    u = np.array([1, 0.8, -0.7, -1])
    assert float_eq(dirichlet_energy(W_PATH, u), u @ laplacian(W_PATH) @ u)


def test_dirichlet_energy_random_symmetric():
    rng = np.random.default_rng(0)
    A = randsym(4, rng)
    u = rng.standard_normal(4)
    assert float_eq(dirichlet_energy(A, u), u @ laplacian(A) @ u)


###
### Laplace learning
###

def test_single_label_spreads_constant():
    # one label on a connected graph: the harmonic extension is constant
    u = laplace_learning(laplacian(W_PATH), np.array([0]), np.array([1.0]))
    assert np.allclose(u, 1.0)


def _two_triangles(bridge):
    # triangles {0,1,2} and {3,4,5} joined by an edge 2-3 of weight `bridge`
    W = np.array([[0, 1, 1, 0, 0, 0],
                  [1, 0, 1, 0, 0, 0],
                  [1, 1, 0, bridge, 0, 0],
                  [0, 0, bridge, 0, 1, 1],
                  [0, 0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 1, 0]], dtype=float)
    L = laplacian(W)
    labeled = np.array([0, 5])
    y = np.array([1.0, -1.0])
    return L, labeled, y, laplace_learning(L, labeled, y)


def test_laplace_learning_two_triangles():
    for bridge in [0.1, 1.0]:
        L, labeled, y, u = _two_triangles(bridge)
        unlabeled = np.setdiff1d(np.arange(6), labeled)
        assert np.allclose(u[labeled], y)              # labels kept
        assert np.allclose((L @ u)[unlabeled], 0)      # harmonic off the labels
        assert np.allclose(u, -u[::-1])                # symmetric setup -> antisymmetric u
        assert np.all(u[:3] > 0) and np.all(u[3:] < 0)  # each triangle follows its label


def test_weak_bridge_gives_more_confident_scores():
    *_, u_weak = _two_triangles(0.1)
    *_, u_strong = _two_triangles(1.0)
    assert abs(u_weak[2]) > abs(u_strong[2])


###
### build_graph
###

def test_build_graph_is_valid_weight_matrix():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    k = 3
    W = build_graph(X, k)
    assert check_diag_sum(W)
    assert check_symmetric(W)
    assert check_min_neighbors(W, k)
    assert np.all(W.data > 0) and np.all(W.data <= 1)
