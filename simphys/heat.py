"""1D heat equation  dT/dt = alpha d2T/dx2 - h (T - T_env), explicit (FTCS) scheme.

Both ends of the bar are held at a fixed temperature.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Material:
    name: str
    conductivity: float  # W / (m K)
    specific_heat: float  # J / (kg K)
    density: float  # kg / m^3

    @property
    def alpha(self) -> float:
        return self.conductivity / (self.specific_heat * self.density)


ALUMINIUM = Material("aluminium", 237.0, 900.0, 2700.0)
IRON = Material("iron", 80.0, 449.0, 7870.0)


def stability_number(alpha, dx, dt):
    """r = alpha dt / dx^2. The explicit scheme needs r <= 1/2 (when h = 0)."""
    return alpha * dt / dx**2


def solve_ftcs(T0, alpha, dx, dt, n_steps, h=0.0, T_env=0.0, left=0.0, right=0.0, store_every=1):
    """March the explicit scheme from the initial profile T0.

    Returns (times, snapshots) where snapshots has shape (len(times), len(T0)).
    """
    T = np.array(T0, dtype=float)
    T[0], T[-1] = left, right
    r = stability_number(alpha, dx, dt)
    times, snaps = [0.0], [T.copy()]
    for step in range(1, n_steps + 1):
        laplacian = T[2:] - 2 * T[1:-1] + T[:-2]
        T[1:-1] = T[1:-1] + r * laplacian - h * dt * (T[1:-1] - T_env)
        if step % store_every == 0:
            times.append(step * dt)
            snaps.append(T.copy())
    return np.array(times), np.array(snaps)


def analytic_uniform(x, t, length, alpha, T0=100.0, n_terms=200):
    """Fourier series for a bar at uniform temperature T0 with both ends clamped at 0."""
    x = np.asarray(x, dtype=float)
    n = 2 * np.arange(1, n_terms + 1) - 1  # only odd modes contribute
    k = n * np.pi / length
    return (np.sin(np.outer(x, k)) * (4 * T0 / (n * np.pi) * np.exp(-alpha * k**2 * t))).sum(axis=1)


def analytic_sine(x, t, length, alpha, amplitude=1.0):
    """Exact solution for a single sine mode: the shape stays, the amplitude decays."""
    x = np.asarray(x, dtype=float)
    return amplitude * np.sin(np.pi * x / length) * np.exp(-alpha * (np.pi / length) ** 2 * t)


def sine_mode_error(nx, r, t_end, length=1.0, alpha=ALUMINIUM.alpha):
    """Max error at t_end for the sine mode on an nx-point grid with stability number r.

    dt is nudged slightly so that an integer number of steps lands exactly on t_end.
    Returns (dx, max_error).
    """
    x = np.linspace(0, length, nx)
    dx = x[1] - x[0]
    n_steps = int(np.ceil(t_end / (r * dx**2 / alpha)))
    dt = t_end / n_steps
    _, snaps = solve_ftcs(
        analytic_sine(x, 0, length, alpha), alpha, dx, dt, n_steps, store_every=n_steps
    )
    return dx, float(np.max(np.abs(snaps[-1] - analytic_sine(x, t_end, length, alpha))))
