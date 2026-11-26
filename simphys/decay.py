"""Monte Carlo radioactive decay: each nucleus decays with probability p per time step."""

import numpy as np


def simulate(n0, p, n_steps, rng=None):
    """Surviving nuclei after each step: array of length n_steps + 1, starting at n0.

    Drawing the number of decays from Binomial(N, p) is equivalent to testing every
    nucleus one by one, and much faster.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = np.empty(n_steps + 1, dtype=np.int64)
    n[0] = n0
    for t in range(n_steps):
        n[t + 1] = n[t] - rng.binomial(n[t], p)
    return n


def fit_decay_rate(n, min_count=1):
    """Least-squares slope of ln N(t) against t, returned as a positive rate.

    Points with fewer than `min_count` survivors are left out (the log amplifies the
    noise of small counts).
    """
    t = np.arange(len(n))
    keep = n >= max(min_count, 1)
    return float(-np.polyfit(t[keep], np.log(n[keep]), 1)[0])


def continuous_rate(p):
    """The rate lambda for which exp(-lambda t) equals (1 - p)^t."""
    return float(-np.log1p(-p))
