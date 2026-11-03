import pytest
import numpy as np

# 4-node path graph: 0 - 1 - 2 - 3
W_PATH = np.array([[0, 1, 0, 0],
                   [1, 0, 1, 0],
                   [0, 1, 0, 1],
                   [0, 0, 1, 0]], dtype=np.float64)

L_PATH = np.array([[ 1, -1,  0,  0],
                   [-1,  2, -1,  0],
                   [ 0, -1,  2, -1],
                   [ 0,  0, -1,  1]], dtype=np.float64)

# graph with six nodes: two triangles {0,1,2}-{3,4,5}
def two_triangles(bridge):
    return np.array([[0, 1, 1, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0],
                     [1, 1, 0, bridge, 0, 0],
                     [0, 0, bridge, 0, 1, 1],
                     [0, 0, 0, 1, 0, 1],
                     [0, 0, 0, 1, 1, 0]], dtype=np.float64)

CASES = {
    "path4": {
        "W": W_PATH,
        "L": L_PATH,                              # Known Laplacian 
        "labeled": np.array([0]),
        "y": np.array([1.0]),
        "u_expected": np.ones(4),                 # one label -> constant
        "energy_check": (np.array([1, 0.8, -0.7, -1]), 2.38),  # (u, energy)
    },
    "triangles_0.1": {
        "W": two_triangles(0.1),
        "labeled": np.array([0, 5]),
        "y": np.array([1.0, -1.0]),
        "u_expected": np.array([1, 16/17, 15/17, -15/17, -16/17, -1]),
    },
    "triangles_1": {
        "W": two_triangles(1.0),
        "labeled": np.array([0, 5]),
        "y": np.array([1.0, -1.0]),
        "u_expected": np.array([1, 5/7, 3/7, -3/7, -5/7, -1]),
    },
}

# A test that takes graph_case as an argument runs once for each case
@pytest.fixture(params=list(CASES))
def graph_case(request):
    return CASES[request.param]

# For tests that test specific cases: pass cases and select a case
@pytest.fixture
def cases():
    return CASES