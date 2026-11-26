"""Unit-step random walks in d dimensions.

Two ways of drawing the step direction are provided:

* "cube":  draw a point uniformly in [-1, 1]^d and normalise it. This is what the
  original project code does. Directions are NOT uniform: diagonals are favoured.
* "uniform": normalise a Gaussian vector, which gives isotropic directions.

Both have zero mean and covariance I/d, so <R^2> = N holds for either one.
"""

import numpy as np


def unit_steps(n, dim, rng, sampler="uniform"):
    if sampler == "uniform":
        raw = rng.standard_normal((n, dim))
    elif sampler == "cube":
        raw = rng.uniform(-1.0, 1.0, (n, dim))
    else:
        raise ValueError(f"unknown sampler: {sampler}")
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def walk(n_steps, dim=2, rng=None, sampler="uniform"):
    """Positions of one walk, starting at the origin: shape (n_steps + 1, dim)."""
    rng = np.random.default_rng() if rng is None else rng
    steps = unit_steps(n_steps, dim, rng, sampler)
    return np.vstack([np.zeros(dim), np.cumsum(steps, axis=0)])


def squared_distances(n_steps, n_walks, dim=2, rng=None, sampler="uniform"):
    """R^2 at the end of each of n_walks independent walks."""
    rng = np.random.default_rng() if rng is None else rng
    out = np.empty(n_walks)
    for i in range(n_walks):
        out[i] = np.sum(unit_steps(n_steps, dim, rng, sampler).sum(axis=0) ** 2)
    return out


def scaling_exponent(n_values, rms_values):
    """Slope of log(R_rms) against log(N). Diffusion gives 1/2."""
    return float(np.polyfit(np.log(n_values), np.log(rms_values), 1)[0])


def cube_angle_density(theta):
    """Exact angle density of the 2D "cube" sampler: 1 / (8 max(cos^2, sin^2))."""
    return 1.0 / (8.0 * np.maximum(np.cos(theta) ** 2, np.sin(theta) ** 2))
