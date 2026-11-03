"""Graph-based semi-supervised learning experiments."""

from .checks import (
    check_diag_sum,
    check_min_neighbors,
    check_symmetric,
    float_eq,
    float_geq,
    float_leq,
)
from .graph import (
    build_graph,
)

from .laplacian import (
    dirichlet_energy,
    laplacian,
    laplacian_sparse,
)

from .solvers import (
    laplace_learning,
    slaplace_learning,
)