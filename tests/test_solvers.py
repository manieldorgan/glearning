import numpy as np
import pytest

from glearning import (
    laplace_learning,
    laplacian,
)


def test_matches_u_expected(graph_case):
    if "labeled" not in graph_case:
        pytest.skip("no labeled indices")   
    elif "y" not in graph_case:
        pytest.skip("no labels y")
    elif "u_expected" not in graph_case:
        pytest.skip("no u_expected")
    
    assert np.allclose(laplace_learning(laplacian(graph_case["W"]), graph_case["labeled"], graph_case["y"]), graph_case["u_expected"])


def test_labels_kept(graph_case):
    if "labeled" not in graph_case:
        pytest.skip("no labeled indices")   
    elif "y" not in graph_case:
        pytest.skip("no labels y")

    L, labeled, y = laplacian(graph_case["W"]), graph_case["labeled"], graph_case["y"]
    assert np.allclose(laplace_learning(L, labeled, y)[labeled], y)    
    

def test_harmonic_at_unlabeled_nodes(graph_case):
    if "labeled" not in graph_case:
        pytest.skip("no labeled indices")   
    elif "y" not in graph_case:
        pytest.skip("no labels y")

    L, labeled, y = laplacian(graph_case["W"]), graph_case["labeled"], graph_case["y"]
    unlabeled = np.setdiff1d(np.arange(L[0].size), labeled)
    u = laplace_learning(L, labeled, y)
    assert np.allclose((L @ u)[unlabeled], 0)


def test_confidence_makes_sense(graph_case):
    if "labeled" not in graph_case:
        pytest.skip("no labeled indices")   
    elif "y" not in graph_case:
        pytest.skip("no labels y")
    
    L, labeled, y = laplacian(graph_case["W"]), graph_case["labeled"], graph_case["y"]
    u = laplace_learning(L, labeled, y)
    assert np.all((u >= np.amin(y) - 1e-12) & (u <= np.amax(y) + 1e-12))


def test_permutation_equivariance():
    pytest.skip("test not yet implemented")



