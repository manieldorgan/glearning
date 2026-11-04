# Two moons & Laplace learning from scratch
## 1. Summary and setup
Laplace learning, implemented from scratch, classifies two moons perfectly from 4 labels and is robust across a wide range of graph settings. It genuinely fails in two ways at the edges: 
- disconnected data clusters when $k$ is too small
- loss of confidence when $\sigma$ is large, especially in combination with large k. 

An apparent third failure for $\sigma \ll1$ turned out to be a numerical error; float64 cannot solve that system. In exact arithmetic the method is perfect there.

**Data.** `make_moons(n_samples=500, noise=0.1, random_state=0)`, labels mapped to $\pm1$.

**Pipeline** (all written and tested by me in this repo).
1. **Graph:** $k$-nearest-neighbor search (Euclidean), Gaussian weights $W_{ij}=\exp(-(d(i,j)/\sigma)^2)$ on kept edges only to avoid `n_samples**2` operations, OR symmetrized.
2. **Laplacian:** $L = D - W$, built sparse.
3. **Solver:** Laplace learning by solving $L_{UU}\,u_U = -L_{U\ell}\,y_\ell$ (dense solve at $n = 500$), with values assembled back into node order.
4. **Prediction:** $\text{sign}(u)$; accuracy measured only on unlabeled points for speedup.

**Checks.**
- **Solver:** reproduces the exact values $16/17$ and $15/17$ on the six-node, two-triangle example, and $5/7$ and $3/7$ with the bridge weight raised to $1$.
- **Laplacian:** matches $D-W$, is symmetric, entries of a row sum to zero, nonnegative eigenvalues, and satisfies the identity $u^TLu=\frac{1}{2}\sum\limits_{j=1}^{n}\sum\limits_{i=1}^{n}W_{ij}(u_{i}-u_{j})^2$. 
- **Graph:** symmetric with a zero diagonal; every run checks that each connected piece of the graph contains a label before solving, and records a failure otherwise.


**Experiments.**

| Experiment      | Settings                                                                                                                                                                              |
| -- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Baseline        | $k = 10$, $\sigma=\text{median of 10-NN distances}=0.089$, 2 labels per class                                                                                                         |
| Parameter sweep | $k \in\{3, 5, 10, 20, 40\}$; $\sigma\in \{0.1, 0.25, 0.5, 1, 2, 4, 10\}\times \text{median(10-NN distances)}$ (fixed reference, so $k$ and $\sigma$ vary independently); 5 seeds each |
| Failure gallery | One representative setting per failure mode                                                                                                                                           |

## 2. Results
### Baseline.
100% accuracy from 4 labels. Confidence $|u|$ is highest deep inside each moon and fades at the tip of the upper moon, where the two moons approach.

### Sweep.
Mean accuracy over 5 seeds:

| Region                                                       | Accuracy                    | Cause                                                                                                  |
| ------------------------------------------------------------ | --------------------------- | ------------------------------------------------------------------------------------------------------ |
| $5\leq k\leq 40$ <br>$\sigma\in[0.25,2]\times \text{median}$ | 0.98–1.00                   | Robust plateau                                                                                         |
| $k = 3$, any $\sigma$                                        | 4 of 5 seeds fail           | Graph splits; a component has no label. The ~0.94 shown is the one connected seed                      |
| $\sigma\in[4,10]\times \text{median}$                        | 0.86–0.96, worst at large k | Far edges become strong; labels leak across, \|u\| shrinks to ~0.3, errors appear where the moons meet |
| $\sigma=0.1\times \text{median}$                             | 0.46–0.54                   | Floating-point failure, not a method failure (see below)                                               |

### Failure Gallery.
**The $k\text{-}\sigma$ interaction.** Large $k$ alone is harmless: at $\sigma = \text{median},\; k = 80$ still scores $1.000$, because far neighbors get negligible weight. Large $k$ hurts only together with large $\sigma$ ($k=80$ at $\sigma=4\times \text{median}$: 0.960). "$k$ too large" and "$\sigma$ too large" are one mechanism: too many effectively strong long-range edges.

**Tiny-$\sigma$ failure is a numerical error.** At $\sigma=0.1\times \text{median}$ ($k = 10$, seed 0), weights range from $0.9$ down to $\approx 10^{-322}$, and the condition number of $L_{UU}$ is $\approx 2\cdot 10^{50}$, far beyond `float64`'s $\approx 16$ digits. The `float64` accuracies are rounding noise:
- The same case gave $0.442$, $0.593$ and $0.562$ across code versions on one machine, and $0.550$ on another.
- Solving the `float64`-rounded matrix exactly gives $0.538$: assembling $L$ already loses the small weights ($10^{-2}+10^{-300}$ rounds to $10^{-2}$).
- Repeating everything (weights, Laplacian, solve) in $4000$-bit interval arithmetic, with every sign certified, gives **100%**, with every u within $10^{-16}$ of $\pm1$. The same code matches the `float64` pipeline to $2.5 \cdot 10^{-14}$ at the baseline $\sigma$.

Why 100% is the correct answer: as $\sigma\to 0$ the strongest edge dominates, so Laplace learning becomes a minimax-path rule. Each point takes the label reachable by the path whose longest step is shortest, and the gap between the moons is wider than any step within a moon.

## 3. Limitations
- **One easy dataset:** points in $\mathbb{R}^2$ and well separated.
- **Small scale:** $n = 500$, dense solves.
- **Few seeds:** $5$ per setting; the $k = 3$ row rests on one connected seed.
- **Narrow scope:** two classes, random fixed labels, Euclidean distance only. No multiclass, active learning, Fermat distance or local scaling yet.
- **Tiny $\sigma$ checked once:** exact arithmetic was run for one setting only; the rest of the $\sigma = 0.1$ column is presumed to be the same artifact.