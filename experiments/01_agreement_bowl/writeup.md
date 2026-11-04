# Agreement bowl
**Summary.** What the cluster assumption implies for two nearby points, and why it is only informative where one class dominates.
 
## Setup
Suppose two nearby points $x$ and $x'$ have (approximately) the same class probabilities $p = (p_1, \dots, p_k)$, and their labels are drawn independently from $p$. Then
 
$$P(\text{same label}) = \sum_{i=1}^{k} p_i^2.$$
 
Since $p$ lies on the probability simplex $\Delta = \{p : p_i \ge 0,\ \sum_i p_i = 1\}$, this quantity satisfies
 
$$\frac{1}{k} \;\leq\; \sum_{i=1}^{k} p_i^2 \;\leq\; 1,$$
 
with the minimum at the center $p = (1/k, \dots, 1/k)$ (fully mixed classes) and the maximum at the corners (one class certain). The lower bound follows from Cauchy–Schwarz: $1 = \left(\sum_i p_i\right)^2 \leq k \sum_i p_i^2$.
 
To plot it, the simplex is projected by dropping the last coordinate, $p_k = 1 - \sum_{i<k} p_i$, and the agreement probability is drawn over the projection:
 
- $k = 2$: the curve $p_1^2 + (1 - p_1)^2$ on $[0, 1]$, minimum $1/2$.
- $k = 3$: the surface $p_1^2 + p_2^2 + (1 - p_1 - p_2)^2$ over a triangle, minimum $1/3$.

![Agreement bowl](figures/agreement_bowl.png)
 
## Takeaway
 
Graph methods assume that strongly connected points share labels. This holds with high probability only near the corners, where one class dominates locally. Near class boundaries the agreement probability drops toward $1/k$, so the assumption carries little information there. This matches the two-moons baseline, where Laplace learning is least confident exactly where the moons come close.
 
## Run
 
```bash
python experiments/02_agreement_bowl/agreement_bowl.py
```
 
Saves `figures/agreement_bowl.png`. No randomness and no package code involved.
