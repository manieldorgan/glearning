import numpy as np
import pytest

from glearning import (
    knn_distances,
    gaussian_weights,
    symmetrize,
    build_graph,
)

@pytest.fixture(scope="module")
def stages():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(250, 2))
    k = 3
    D = knn_distances(X, k)
    sigma = np.median(D.data)
    W_raw = gaussian_weights(D, sigma)
    W = symmetrize(W_raw)
    return {"X": X, "k": k, "D": D, "sigma": sigma, "W_raw": W_raw, "W": W}
    

def test_diag_sum(stages):
    assert np.all(stages["W"].diagonal() == pytest.approx(0.0))


def test_W_is_symmetric(stages):
    W = stages["W"]
    assert np.abs(W - W.T).max() == pytest.approx(0.0)


def test_exactly_knn_per_row(stages):
    assert np.all(np.diff(stages["D"].indptr) == stages["k"])


def test_min_neighbors(stages):
    W = stages["W"].copy() 
    W.eliminate_zeros()
    assert bool(np.all(np.diff(W.indptr) >= stages["k"]))


def test_weights_make_sense(stages):
    W = stages["W"].copy() 
    W.eliminate_zeros()
    assert np.all(W.data > 0) and np.all(W.data <= 1)

