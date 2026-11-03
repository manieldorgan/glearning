import numpy as np

# Creates a random symmetric numpy array of size n
# sample drawn from the standard normal distribution
def randsym(n, rng):
    A = rng.standard_normal((n, n))
    return (A + A.T) / 2

# Direct computation of the dirichlet energy
def dirichlet_energy(W, u):
    return 0.5 * (W * (np.subtract.outer(u, u)) ** 2).sum(axis=1).sum(axis=0)