import numpy as np
import pytest

from simphys import heat, ode


def final_error(step, n):
    osc = ode.Oscillator()
    t, y = ode.integrate(step, osc.rhs, [0.0, 1.0], 0.0, 2.0, n)
    return np.max(np.abs(y[:, 0] - osc.exact(t, 0.0, 1.0)[:, 0]))


@pytest.mark.parametrize("step, order", [(ode.rk2_step, 2), (ode.rk4_step, 4), (ode.verlet_step, 2)])
def test_convergence_order(step, order):
    observed = np.log2(final_error(step, 400) / final_error(step, 800))
    assert observed == pytest.approx(order, abs=0.15)


def test_exact_solution_satisfies_initial_conditions():
    y = ode.Oscillator().exact([0.0], 3.0, -5.0)
    np.testing.assert_allclose(y[0], [3.0, -5.0])


def test_adaptive_solver_reaches_end_and_is_accurate():
    osc = ode.Oscillator()
    t, y, _ = ode.rkf45(osc.rhs, [0.0, 1.0], 0.0, 10.0, tol=1e-8)
    assert t[-1] == pytest.approx(10.0)
    assert np.max(np.abs(y[:, 0] - osc.exact(t, 0.0, 1.0)[:, 0])) < 1e-5


def test_verlet_energy_stays_bounded_while_rk2_drifts():
    osc = ode.Oscillator()
    drift = {}
    for name, step in [("rk2", ode.rk2_step), ("verlet", ode.verlet_step)]:
        _, y = ode.integrate(step, osc.rhs, [0.0, 1.0], 0.0, 50.0, 1000)
        drift[name] = np.max(np.abs(osc.energy(y) / osc.energy(y[0]) - 1))
    assert drift["verlet"] < 0.2 < drift["rk2"]


def test_period_is_independent_of_amplitude():
    osc = ode.Oscillator()
    periods = []
    for v0 in (0.5, 8.0):
        t, y = ode.integrate(ode.rk4_step, osc.rhs, [0.0, v0], 0.0, 10.0, 1000)
        periods.append(ode.zero_crossing_period(t, y[:, 0]))
    assert periods == pytest.approx([1.0, 1.0], abs=1e-4)


def test_ftcs_matches_fourier_series():
    x = np.linspace(0, 1, 101)
    alpha = heat.ALUMINIUM.alpha
    times, snaps = heat.solve_ftcs(np.full(101, 100.0), alpha, 0.01, 0.45, 2000, store_every=2000)
    exact = heat.analytic_uniform(x, times[-1], 1.0, alpha)
    assert np.max(np.abs(snaps[-1] - exact)) < 0.5


def test_ftcs_is_second_order_in_space():
    dx1, e1 = heat.sine_mode_error(21, r=0.4, t_end=500.0)
    dx2, e2 = heat.sine_mode_error(81, r=0.4, t_end=500.0)
    assert np.log(e1 / e2) / np.log(dx1 / dx2) == pytest.approx(2.0, abs=0.1)


def test_r_one_sixth_is_fourth_order():
    dx1, e1 = heat.sine_mode_error(11, r=1 / 6, t_end=500.0)
    dx2, e2 = heat.sine_mode_error(41, r=1 / 6, t_end=500.0)
    assert np.log(e1 / e2) / np.log(dx1 / dx2) > 3.5


@pytest.mark.parametrize("r, stable", [(0.49, True), (0.52, False)])
def test_stability_limit(r, stable):
    alpha, dx = heat.ALUMINIUM.alpha, 0.01
    _, snaps = heat.solve_ftcs(np.full(101, 100.0), alpha, dx, r * dx**2 / alpha, 2000, store_every=2000)
    assert bool(np.max(np.abs(snaps[-1])) <= 100.0) == stable


def test_cooling_term_lowers_temperature():
    alpha = heat.ALUMINIUM.alpha
    start = np.full(101, 100.0)
    _, insulated = heat.solve_ftcs(start, alpha, 0.01, 0.45, 1000)
    _, cooled = heat.solve_ftcs(start, alpha, 0.01, 0.45, 1000, h=1e-3)
    assert np.all(cooled[-1][1:-1] < insulated[-1][1:-1])
