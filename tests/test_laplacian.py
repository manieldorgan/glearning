import numpy as np
import pytest

from helpers import randsym

from glearning import (
    dirichlet_energy,
    laplacian,
    laplacian_sparse,
)



###
### Direct Laplacian Checks
###

def test_laplacian_known_value(graph_case):
    if "L" not in graph_case:
        pytest.skip("no known Laplacian")
    assert laplacian(graph_case["W"]) == pytest.approx(graph_case["L"])


def test_laplacian_kills_constants(graph_case):
    W = graph_case["W"]
    assert laplacian(W) @ np.ones(len(W)) == pytest.approx(0)


def test_laplacian_sparse_matches_dense(graph_case):
    assert laplacian_sparse(graph_case["W"]).toarray() == pytest.approx(laplacian(graph_case["W"])), "graph Laplacian is wrong"



###
### Dirichlet energy <--> (u^T)Lu comparison checks
###

def test_dirichlet_energy_known_value(graph_case):
    if "energy_check" not in graph_case:
        pytest.skip("no known energy")
    u, expected = graph_case["energy_check"]     # tuple unpacking
    assert dirichlet_energy(graph_case["W"], u) == pytest.approx(expected)


def test_dirichlet_energy_equals_quadratic_form(graph_case):
    rng = np.random.default_rng(0)
    W = graph_case["W"]
    u = rng.standard_normal((len(W), ))
    assert dirichlet_energy(W, u) == pytest.approx(u @ laplacian(W) @ u)
    

def test_dirichlet_energy_random_symmetric():
    rng = np.random.default_rng(0)
    A = randsym(4, rng)
    u = rng.standard_normal(4)
    assert dirichlet_energy(A, u) == pytest.approx(u @ laplacian(A) @ u)

