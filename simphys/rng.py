"""Linear congruential generators and a few checks of their quality."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LCG:
    """r_{i+1} = (a r_i + c) mod m, returned as r / m in [0, 1)."""

    a: int
    c: int
    m: int
    seed: int = 10

    def integers(self, n):
        r, out = self.seed, np.empty(n, dtype=np.uint64)
        for i in range(n):
            r = (self.a * r + self.c) % self.m
            out[i] = r
        return out

    def uniforms(self, n):
        return self.integers(n).astype(float) / self.m

    def period(self, limit=10**6):
        """Length of the cycle reached from the seed (None if longer than `limit`)."""
        r, seen = self.seed, {}
        for i in range(limit):
            if r in seen:
                return i - seen[r]
            seen[r] = i
            r = (self.a * r + self.c) % self.m
        return None


TOY = LCG(a=57, c=1, m=256)  # deliberately poor
GOOD = LCG(a=0x5DEECE66D, c=0xB, m=2**48)  # the constants used by drand48 and java.util.Random
RANDU = LCG(a=65539, c=0, m=2**31, seed=1)  # infamous IBM generator


def moments(u, orders=(1, 2, 3, 4, 5)):
    """Sample moments <u^k>. A uniform variable has 1 / (k + 1)."""
    return {k: float(np.mean(u**k)) for k in orders}


def lag_correlation(u, lag):
    """C(k) = mean of u_i u_{i+k}. Independent uniforms give 1/4."""
    return float(np.mean(u[:-lag] * u[lag:]))


def pair_chi_square(u, bins=16):
    """Chi-square statistic of non-overlapping pairs on a bins x bins grid.

    For a good generator it is close to its degrees of freedom, bins**2 - 1.
    """
    n = len(u) // 2
    counts, _, _ = np.histogram2d(u[0 : 2 * n : 2], u[1 : 2 * n : 2], bins=bins, range=[[0, 1], [0, 1]])
    expected = n / bins**2
    return float(((counts - expected) ** 2 / expected).sum())


def randu_plane_index(integers):
    """For RANDU,  9 x_i - 6 x_{i+1} + x_{i+2}  is an exact multiple of 2^31.

    Returns that multiple for every consecutive triple. Only a handful of distinct values
    exist, so all triples lie on that many parallel planes.
    """
    x = np.asarray(integers, dtype=np.int64)
    combo = 9 * x[:-2] - 6 * x[1:-1] + x[2:]
    if np.any(combo % 2**31):
        raise ValueError("sequence does not satisfy the RANDU recurrence")
    return combo // 2**31
