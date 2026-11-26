"""Explicit integrators for the harmonic oscillator  m x'' = -k x."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Oscillator:
    m: float = 1.0
    k: float = (2 * np.pi) ** 2  # period T = 1

    @property
    def omega(self) -> float:
        return float(np.sqrt(self.k / self.m))

    def rhs(self, t, y):
        x, v = y
        return np.array([v, -(self.k / self.m) * x])

    def exact(self, t, x0, v0):
        """Analytic position and velocity, shape (len(t), 2)."""
        t = np.asarray(t, dtype=float)
        w = self.omega
        x = x0 * np.cos(w * t) + (v0 / w) * np.sin(w * t)
        v = -x0 * w * np.sin(w * t) + v0 * np.cos(w * t)
        return np.stack([x, v], axis=-1)

    def energy(self, y):
        y = np.asarray(y)
        return 0.5 * self.m * y[..., 1] ** 2 + 0.5 * self.k * y[..., 0] ** 2


def rk2_step(f, t, y, h):
    """Midpoint method (second order)."""
    k1 = f(t, y)
    return y + h * f(t + h / 2, y + h / 2 * k1)


def rk4_step(f, t, y, h):
    """Classical fourth-order Runge-Kutta."""
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def verlet_step(f, t, y, h):
    """Velocity Verlet for a position-only force (second order, symplectic)."""
    x, v = y
    a0 = f(t, y)[1]
    x_new = x + h * v + h**2 / 2 * a0
    a1 = f(t + h, np.array([x_new, v]))[1]
    return np.array([x_new, v + h / 2 * (a0 + a1)])


def integrate(step, f, y0, t0, t1, n):
    """Fixed-step integration with n steps. Returns (t, y) with y of shape (n+1, dim)."""
    t = np.linspace(t0, t1, n + 1)
    h = (t1 - t0) / n
    y = np.empty((n + 1, len(y0)))
    y[0] = y0
    for i in range(n):
        y[i + 1] = step(f, t[i], y[i], h)
    return t, y


# Fehlberg 4(5) tableau
_C = (0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2)
_A = (
    (),
    (1 / 4,),
    (3 / 32, 9 / 32),
    (1932 / 2197, -7200 / 2197, 7296 / 2197),
    (439 / 216, -8, 3680 / 513, -845 / 4104),
    (-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40),
)
_B4 = np.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0])
_B5 = np.array([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])


def rkf45(f, y0, t0, t1, tol=1e-8, h0=0.05, h_min=1e-6, h_max=0.2):
    """Adaptive Runge-Kutta-Fehlberg. A step is accepted when the embedded
    4th/5th order difference is below `tol` (infinity norm).

    Returns (t, y, number_of_rejected_steps).
    """
    t, y, h = float(t0), np.asarray(y0, dtype=float), float(h0)
    ts, ys, rejected = [t], [y.copy()], 0
    while t1 - t > 1e-12:
        h = min(h, t1 - t)
        k = []
        for c, row in zip(_C, _A):
            increment = sum((a * ki for a, ki in zip(row, k)), np.zeros_like(y))
            k.append(f(t + c * h, y + h * increment))
        k = np.array(k)
        y5 = y + h * (_B5 @ k)
        err = np.max(np.abs(y5 - (y + h * (_B4 @ k))))
        if err <= tol or h <= h_min * 1.001:
            t, y = t + h, y5
            ts.append(t)
            ys.append(y.copy())
        else:
            rejected += 1
        scale = 2.0 if err == 0 else 0.9 * (tol / err) ** 0.2
        h = min(h_max, max(h_min, h * min(5.0, max(0.2, scale))))
    return np.array(ts), np.array(ys), rejected


def zero_crossing_period(t, x):
    """Mean time between upward zero crossings (linear interpolation)."""
    i = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
    if len(i) < 2:
        return float("nan")
    crossings = t[i] - x[i] * (t[i + 1] - t[i]) / (x[i + 1] - x[i])
    return float(np.mean(np.diff(crossings)))
