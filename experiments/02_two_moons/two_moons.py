import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csgraph
from sklearn.datasets import make_moons
from sklearn.neighbors import kneighbors_graph

import glearning as glr

PLOTS_DIR = Path(__file__).parent.resolve() / "figures"

###
### Experiment Production Stuff
###

def pick_labels(true_labels, n_per_class, rng): 
    # choosing n random indices for labeled points in each class (+1,-1)
    plus = rng.choice(np.where(true_labels == 1)[0], size=n_per_class, replace=False) 
    neg = rng.choice(np.where(true_labels == -1)[0], size=n_per_class, replace=False)

    labeled = np.concatenate((plus, neg))
    y=true_labels[labeled]
    
    return labeled, y

def median_knn_distance(X, k): 
    return np.median(kneighbors_graph(X, k, mode='distance').data)

def run_experiment(X, true_labels, k, sigma, n_per_class, seed):
    rng = np.random.default_rng(seed) 

    result = {
        "u" : None,
        "pred" : None,
        "acc" : np.nan,
        "labeled" : None,
        "n_comp" : None,
        "comp" : None,
        "ok" : None,
        "k" : k,
        "sigma" : sigma
        }
    
    result["labeled"], y = pick_labels(true_labels, n_per_class, rng)
    
    W = glr.build_graph(X, k, sigma)
    W.eliminate_zeros()

    result["n_comp"], result["comp"] = csgraph.connected_components(W, directed=False) 
    result["ok"] = set(result["comp"][result["labeled"]]) == set(range(result["n_comp"]))
    if not result["ok"]:
        return result

    L = glr.laplacian_sparse(W).toarray()
    result["u"] = glr.laplace_learning(L, result["labeled"], y)
    result["pred"] = np.sign(result["u"])
    
    mask = np.ones(result["u"].size, dtype=bool)
    mask[result["labeled"]] = False
    result["acc"]= np.mean(result["pred"][mask] == true_labels[mask])

    return result


###
### Plotting Stuff
###

def _style(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _mark_labeled(ax, X, labeled):
    ax.scatter(X[labeled, 0], X[labeled, 1], s=220, marker="*", c="gold",
               edgecolors="black", linewidths=1.2, zorder=3, label="labeled")


def plot_baseline(X, true_labels, base, fname=PLOTS_DIR / "baseline.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    kw = dict(s=14, cmap="coolwarm", vmin=-1, vmax=1)

    axes[0].scatter(X[:, 0], X[:, 1], c=true_labels, **kw)
    axes[0].set_title("True labels")

    axes[1].scatter(X[:, 0], X[:, 1], c=base["pred"], **kw)
    unlabeled = np.ones(len(X), dtype=bool)
    unlabeled[base["labeled"]] = False
    wrong = unlabeled & (base["pred"] != true_labels)
    axes[1].scatter(X[wrong, 0], X[wrong, 1], s=70, facecolors="none",
                    edgecolors="black", linewidths=1.0, label="misclassified")
    axes[1].set_title(f"Prediction sign(u), accuracy {base['acc']:.3f}")

    sc = axes[2].scatter(X[:, 0], X[:, 1], c=base["u"], **kw)
    fig.colorbar(sc, ax=axes[2], label="u")
    axes[2].set_title("Score u (strength = confidence)")

    for ax in axes:
        _mark_labeled(ax, X, base["labeled"])
        _style(ax)
    axes[1].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Laplace learning on two moons: k = {base['k']}, "
                 f"σ = {base['sigma']:.3f}, {len(base['labeled'])} labels")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    return fig


def plot_sweep(acc_grid, ks, mults, fname=PLOTS_DIR / "sweep.png"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN cells
        mean_acc = np.nanmean(acc_grid, axis=2)
    fail_frac = np.isnan(acc_grid).mean(axis=2)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(mean_acc, cmap=cmap, vmin=0.5, vmax=1.0,
                   origin="lower", aspect="auto")
    fig.colorbar(im, ax=ax, label="mean accuracy (unlabeled points)")

    ax.set_xticks(range(len(mults)))
    ax.set_xticklabels([str(m) for m in mults])
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([str(k) for k in ks])
    ax.set_xlabel("σ / median 10-NN distance")
    ax.set_ylabel("k (neighbors)")

    for i in range(len(ks)):
        for j in range(len(mults)):
            if np.isnan(mean_acc[i, j]):
                text = "fail"
            else:
                text = f"{mean_acc[i, j]:.2f}"
                if fail_frac[i, j] > 0:
                    text += f"\n({fail_frac[i, j]:.0%} fail)"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if not np.isnan(mean_acc[i, j]) and mean_acc[i, j] < 0.8 else "black")

    ax.set_title(f"Accuracy over k and σ (mean of {acc_grid.shape[2]} seeds; gray = graph split, no solve)")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    return fig


def plot_gallery(X, gallery, fname=PLOTS_DIR / "gallery.png"):
    fig, axes = plt.subplots(1, len(gallery), figsize=(18, 4.5), layout="constrained")
    last_u = None
    for ax, (title, res) in zip(axes, gallery):
        if res["u"] is not None:
            last_u = ax.scatter(X[:, 0], X[:, 1], c=res["u"], s=14,
                                cmap="coolwarm", vmin=-1, vmax=1)
            sub = f"accuracy {res['acc']:.3f}"
        elif res["comp"] is not None:
            ax.scatter(X[:, 0], X[:, 1], c=res["comp"], s=14, cmap="tab20")
            sub = f"{res['n_comp']} pieces; a piece has no label, no solve"
        else:
            sub = "build failed"
        _mark_labeled(ax, X, res["labeled"])
        _style(ax)
        ax.set_title(f"{title}\nk = {res['k']}, σ = {res['sigma']:.3f}\n{sub}", fontsize=10)
    if last_u is not None:
        fig.colorbar(last_u, ax=axes, label="u", shrink=0.8)
    fig.suptitle("degenerate cases: colored by u, or by connected piece when the graph splits")
    fig.savefig(fname, dpi=150)
    return fig


###
### Main Script
###

if __name__ == "__main__":
    # Setup
    seed = 0
    X, true_labels = make_moons(n_samples=500, noise=0.1, random_state=seed)
    true_labels = 2*true_labels - 1  # map (0,1) -> (-1,+1) for solver

    # Record baseline
    med10 = median_knn_distance(X, 10)
    base = run_experiment(X, true_labels, k=10, sigma=1.0 * med10, n_per_class=2, seed=0)

    # Record results for various parameter combos
    ks = [3, 5, 10, 20, 40]
    mults = [0.1, 0.25, 0.5, 1, 2, 4, 10]
    seeds = range(5)
    acc_grid = np.array([[[run_experiment(X, true_labels, k=a, sigma=b*med10, n_per_class=2, seed=c)["acc"]  for c in seeds]  for b in mults] for a in ks])

    # Record degenerate cases
    gallery = [
        ("k too small", run_experiment(X, true_labels, k=3, sigma=1.0 * med10, n_per_class=2, seed=3)),
        ("sigma too small", run_experiment(X, true_labels, k=10, sigma=0.1 * med10, n_per_class=2, seed=0)),
        ("k too large", run_experiment(X, true_labels, k=80, sigma=4 * med10, n_per_class=2, seed=0)),
        ("sigma too large", run_experiment(X, true_labels, k=40, sigma=10 * med10, n_per_class=2, seed=0))
        ]

    # Plot plots
    PLOTS_DIR.mkdir(exist_ok=True)
    plot_baseline(X, true_labels, base)
    plot_sweep(acc_grid, ks, mults)
    plot_gallery(X, gallery)
    plt.show()

