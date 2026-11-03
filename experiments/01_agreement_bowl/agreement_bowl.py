import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from pathlib import Path

# plots/ at the repo root, no matter where the script is run from
PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots"


def simplex_points(k, m):
    """Grid of points on the probability simplex in R^k (k = 2 or 3), spacing 1/m.
    Returns an (N, k) array; every row is nonnegative and sums to 1."""
    if k == 2:
        p1 = np.linspace(0, 1, m + 1)
        P = np.column_stack([p1, 1 - p1])
    elif k == 3:
        P = np.array([(i / m, j / m, 1 - (i + j) / m)
                      for i in range(m + 1) for j in range(m + 1 - i)])
    else:
        raise ValueError("only k = 2 or 3 can be drawn")
    return np.clip(P, 0, 1)            

# Drop the last coordinate
def project(P):
    return P[:, :-1]

# Compute local agreement probability
def agreement(P):
    return (P ** 2).sum(axis=1)

# k = 2
P2 = simplex_points(2, 250)       
Q2 = project(P2)                  
F2 = agreement(P2)

# k = 3
P3 = simplex_points(3, 60)        
Q3 = project(P3)                  
F3 = agreement(P3)
tri = mtri.Triangulation(Q3[:, 0], Q3[:, 1])   # one mesh for both surfaces

fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(2, 2)

# Panel 1: k = 2 simplex and its projection
ax = fig.add_subplot(gs[0, 0])
ax.plot(P2[:, 0], P2[:, 1], color="tab:orange", lw=3, label="simplex: $p_1 + p_2 = 1$")
ax.plot(Q2[:, 0], np.zeros(len(Q2)), color="tab:blue", lw=3, label="projection onto $p_1$-axis")
for p1 in [0.0, 0.25, 0.5, 0.75, 1.0]:
    ax.plot([p1, p1], [1 - p1, 0], "--", color="gray", lw=1)
ax.set_xlabel("$p_1$"); ax.set_ylabel("$p_2$")
ax.set_xlim(-0.05, 1.1); ax.set_ylim(-0.05, 1.1); ax.set_aspect("equal")
ax.set_title("k = 2: the simplex and its projection")
ax.legend(loc="upper right"); ax.grid(True)

# Panel 2: k = 2 bowl over the projection
ax = fig.add_subplot(gs[0, 1])
ax.plot(Q2[:, 0], F2, color="tab:green", lw=2, label="P(same label)")
ax.plot(Q2[:, 0], np.full(len(Q2), 0.42), color="tab:blue", lw=3, label="domain: the projection [0, 1]")
ax.axhline(0.5, color="gray", ls=":", lw=1)
ax.set_xlabel("$p_1$"); ax.set_ylabel("P(same label)")
ax.set_ylim(0.38, 1.05)
ax.set_title("k = 2 bowl: " r"$p_1^2 + (1-p_1)^2$" "\nminimum 1/2 at the center, 1 at the corners")
ax.legend(loc="upper center"); ax.grid(True)

# Panel 3: k = 3 simplex in R^3 and its projection
ax = fig.add_subplot(gs[1, 0], projection="3d")
ax.plot_trisurf(P3[:, 0], P3[:, 1], P3[:, 2], triangles=tri.triangles,
                color="tab:orange", alpha=0.6)
ax.plot_trisurf(Q3[:, 0], Q3[:, 1], np.zeros(len(Q3)), triangles=tri.triangles,
                color="tab:blue", alpha=0.35)
corners = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1/3, 1/3, 1/3]])
for c in corners:
    ax.plot([c[0], c[0]], [c[1], c[1]], [c[2], 0], "--", color="gray", lw=1)
ax.set_xlabel("$p_1$"); ax.set_ylabel("$p_2$"); ax.set_zlabel("$p_3$")
ax.set_title("k = 3: simplex (orange, in $\\mathbb{R}^3$)\nand its projection (blue, $p_3$ dropped)")
ax.view_init(elev=25, azim=35)

# Panel 4: k = 3 bowl over the projection
ax = fig.add_subplot(gs[1, 1], projection="3d")
bowl = ax.plot_trisurf(Q3[:, 0], Q3[:, 1], F3, triangles=tri.triangles, cmap="viridis")
floor = 0.25
edge = np.array([[0, 0], [1, 0], [0, 1], [0, 0]])
ax.plot(edge[:, 0], edge[:, 1], np.full(4, floor), color="tab:blue", lw=2.5,
        label="domain: the projection")
ax.set_zlim(floor, 1.0)
ax.set_xlabel("$p_1$"); ax.set_ylabel("$p_2$"); ax.set_zlabel("P(same label)")
ax.set_title("k = 3 bowl: " r"$p_1^2 + p_2^2 + (1-p_1-p_2)^2$" "\nminimum 1/3 at the center, 1 at the corners")
ax.legend(loc="upper right")
fig.colorbar(bowl, ax=ax, shrink=0.6, pad=0.12, label="P(same label)")

fig.tight_layout()
PLOTS_DIR.mkdir(exist_ok=True)
fig.savefig(PLOTS_DIR / "agreement_bowl.png", dpi=150)
plt.show()

