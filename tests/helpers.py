

# Creates a random symmetric numpy array of size n
# sample drawn from the standard normal distribution
def randsym(n, rng):
    A = rng.standard_normal((n, n))
    return (A + A.T) / 2