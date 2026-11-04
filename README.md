# glearning
 
Graph-based semi-supervised learning, implemented from scratch in NumPy/SciPy, with each addition validated, tested, and written up as an experiment.
 
**Status (Nov 2026):** binary Laplace learning on Euclidean $k$-NN graphs, with a test suite and two experiments. Validation against [GraphLearning](https://github.com/jwcalder/GraphLearning), sparse solvers, multiclass, Poisson learning, Fermat-distance graphs and active learning are planned, in that order.
 
## Install
 
```bash
conda env create -f environment.yml
conda activate glearning
pip install -e .        # editable install, so code changes take effect immediately
pytest                  # everything should pass
```
 
## Quick example
 
```python
import numpy as np
from scipy.sparse import csgraph
from sklearn.datasets import make_moons
import glearning as glr
 
X, t = make_moons(n_samples=500, noise=0.1, random_state=0)
t = 2 * t - 1                                  # labels in {-1, +1}
 
W = glr.build_graph(X, k=10)                   # sigma defaults to the median k-NN distance
W.eliminate_zeros()
n_comp, comp = csgraph.connected_components(W, directed=False)
 
rng = np.random.default_rng(0)
labeled = np.concatenate([rng.choice(np.where(t == s)[0], 2, replace=False) for s in (1, -1)])
assert set(comp[labeled]) == set(range(n_comp)), "every component needs a label"
 
L = glr.laplacian_sparse(W).toarray()          # dense solve for now
u = glr.laplace_learning(L, labeled, t[labeled])
pred = np.sign(u)
```
 
## The math
 
**Graph.** Each point $x_i$ is joined to its $k$ nearest neighbors, with Gaussian weights
 
$$W_{ij} = \exp\left(-\frac{\lVert x_i - x_j \rVert^2}{\sigma^2}\right),$$
 
and the graph is symmetrized by keeping an edge if either endpoint chose it: $W_{ij}= \max(W_{ij}, W_{ji})$.
 
**Laplacian.** $L = D - W$, where $D = \operatorname{diag}(d_1, \dots, d_n)$ and $d_i = \sum_j W_{ij}$. For any $u \in \mathbb{R}^n$,
 
$$u^\top L u = \frac{1}{2} \sum_{i,j} W_{ij} (u_i - u_j)^2,$$
 
which is small exactly when $u$ varies little across strong edges.
 
**Laplace learning.** Given labels $y_\ell$ on a labeled set $\ell$, find $u$ minimizing $u^\top L u$ subject to $u_\ell = y_\ell$. Splitting $L$ into labeled and unlabeled blocks, the minimizer solves
 
$$L_{UU}\, u_U = -L_{U\ell}\, y_\ell,$$
 
and predictions are $\operatorname{sign}(u)$. The system has a unique solution when every connected component of the graph contains a label.
 
**Numerical caveat.** When weights span many orders of magnitude (very small $\sigma$), $L_{UU}$ can be so ill-conditioned that float64 results are meaningless. As a rule of thumb, a solve keeps about $16 - \log_{10} \kappa(L_{UU})$ trustworthy digits. See the two-moons write-up for a case with $\kappa \approx 2 \times 10^{50}$.
 
## Layout
 
```
src/glearning/
    __init__.py     # public API: everything users import comes from here
    graph.py        # knn_distances -> gaussian_weights -> symmetrize; build_graph(metric=...)
    laplacian.py    # laplacian, laplacian_sparse
    solvers.py      # laplace_learning
tests/
    conftest.py     # shared graph cases (CASES) and fixtures
    helpers.py      # test-only utilities
    test_graph.py, test_laplacian.py, test_solvers.py   # one per module
experiments/
    01_two_moons/       # Laplace learning on two moons: baseline, k-sigma sweep, failure modes
    02_agreement_bowl/  # what the cluster assumption implies for two nearby points
```
 
## Extending the package
 
Most of my additions follow the same workflow, on their own branch:
 
1. **Understand first.** Be able to derive the method on paper before coding it.
2. **Implement** it in the module it belongs to, or a new one, and export it from `__init__.py`.
3. **Test** it in the matching `tests/test_<module>.py`:
   - *Property tests* take the `graph_case` fixture and run on every case in `CASES`. Examples: $L\mathbf{1} = 0$, labels are kept, $u$ is harmonic at unlabeled nodes, and $\min y \le u_i \le \max y$.
   - *Known-value tests* compare against an answer worked out by hand and stored in the case (for example `"u_expected"`), and skip cases without one.
   - To add a test graph, add an entry to `CASES` in `tests/conftest.py`. Every property test then covers it automatically.
4. **Check** that `pytest` still passes and that `validate.py` still matches GraphLearning.
5. **Run an experiment** that shows the weakness being addressed, the fix, and the improvement, with seeds and settings recorded. Check $\kappa(L_{UU})$ before trusting results in extreme regimes.
6. **Write it up** in `experiments/NN_name/README.md`.
7. **Commit** code, figures and write-up together.

### Conventions
 
- **Sparse arrays:** I use SciPy's `csr_array`, not the older `csr_matrix`.
- **No hidden mutation:** functions NEVER modify their inputs (unless that is their sole purpose).
- **Randomness:** package functions take `rng=None` and start with `rng = np.random.default_rng(rng)`. Seeds are chosen only in scripts and tests, one per run; a run with several random processes splits its seed with `np.random.default_rng(seed).spawn(n)`.
- **Imports:** each module imports what it uses; tests and experiments import only from `glearning`, never from submodules.

## Experiments
 
| | Experiment | Finding |
| --- | --- | --- |
| 01 | [Two moons](experiments/01_two_moons/) | 100% from 4 labels; robust over a wide $(k, \sigma)$ range; fails by disconnection (small $k$) and label leakage (large $\sigma$). The small-$\sigma$ "failure" is float64 rounding: exact arithmetic gives 100%. |
| 02 | [Agreement bowl](experiments/02_agreement_bowl/) | Two nearby points share a label with probability $\sum_i p_i^2$, between $1/k$ and $1$: the smoothness assumption is only informative where one class dominates. |
 
## Roadmap
 
1. Real datasets (MNIST subsets) and a validation suite against GraphLearning
2. Sparse direct and conjugate-gradient solvers
3. Multiclass Laplace learning
4. Poisson learning
5. Low-label-rate behavior as dimension grows (Laplace vs. Poisson)
6. Fermat-distance graphs
7. Active learning, then reproducing a MALADY result
## References
 
- J. Calder, *GraphLearning* Python package: <https://github.com/jwcalder/GraphLearning>
- J. Calder, B. Cook, M. Thorpe, D. Slepčev, *Poisson Learning: Graph-Based Semi-Supervised Learning at Very Low Label Rates*, ICML 2020. [arXiv:2006.11184](https://arxiv.org/abs/2006.11184)