"""Run every experiment, write the figures to figures/ and the numbers to results/summary.json.

    python run_all.py              # everything
    python run_all.py heat decay   # selected sections
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

from simphys import decay, heat, ode, random_walk, rng

FIGURES = Path(__file__).parent / "figures"
RESULTS = Path(__file__).parent / "results"

BLUE, ORANGE, AQUA, YELLOW = "#257de0", "#f47230", "#00b894", "#f2a600"
INK, MUTED = "#191919", "#666666"
SEQUENTIAL = plt.get_cmap("Blues")

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "lines.linewidth": 2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.7,
        "axes.edgecolor": MUTED,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "font.size": 10,
    }
)


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)


# --------------------------------------------------------------------------- ODEs
def run_ode():
    osc = ode.Oscillator()
    methods = {"RK2": (ode.rk2_step, BLUE), "RK4": (ode.rk4_step, ORANGE), "Verlet": (ode.verlet_step, AQUA)}
    out = {"max_position_error_n100": {}, "observed_order": {}, "max_relative_energy_error_t50": {}}

    # error over time on the coarse grid used by the project (h = 0.1, ten periods)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for name, (step, color) in methods.items():
        t, y = ode.integrate(step, osc.rhs, [0.0, 1.0], 0.0, 10.0, 100)
        err = np.abs(y[:, 0] - osc.exact(t, 0.0, 1.0)[:, 0])
        out["max_position_error_n100"][name] = float(err.max())
        ax.semilogy(t[1:], err[1:], color=color, label=name)
    ax.set(xlabel="time (periods)", ylabel="|x - exact|", title="Position error, h = 0.1")
    ax.legend(ncol=3)
    save(fig, "ode_error_vs_time")

    # convergence
    fig, ax = plt.subplots(figsize=(6, 4))
    ns = np.array([100, 200, 400, 800, 1600, 3200])
    for name, (step, color) in methods.items():
        errors = []
        for n in ns:
            t, y = ode.integrate(step, osc.rhs, [0.0, 1.0], 0.0, 2.0, n)
            errors.append(np.max(np.abs(y[:, 0] - osc.exact(t, 0.0, 1.0)[:, 0])))
        h = 2.0 / ns
        slope = float(np.polyfit(np.log(h[2:]), np.log(errors[2:]), 1)[0])
        out["observed_order"][name] = slope
        ax.loglog(h, errors, "o-", color=color, label=f"{name} (slope {slope:.2f})")
    ax.set(xlabel="step size h", ylabel="max error over two periods", title="Convergence order")
    ax.legend()
    save(fig, "ode_convergence")

    # energy
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for name, (step, color) in methods.items():
        t, y = ode.integrate(step, osc.rhs, [0.0, 1.0], 0.0, 50.0, 1000)
        rel = osc.energy(y) / osc.energy(y[0]) - 1
        out["max_relative_energy_error_t50"][name] = float(np.max(np.abs(rel)))
        ax.plot(t, rel, color=color, label=name)
    ax.set(xlabel="time (periods)", ylabel="E(t) / E(0) - 1", title="Energy drift over 50 periods, h = 0.05")
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.legend(ncol=3)
    save(fig, "ode_energy")

    # adaptive step
    t, y, rejected = ode.rkf45(osc.rhs, [0.0, 1.0], 0.0, 10.0, tol=1e-8)
    out["rkf45"] = {
        "accepted_steps": len(t) - 1,
        "rejected_steps": rejected,
        "max_position_error": float(np.max(np.abs(y[:, 0] - osc.exact(t, 0.0, 1.0)[:, 0]))),
    }

    # period against amplitude (isochronism)
    periods = {}
    for v0 in (0.5, 1.0, 2.0, 4.0, 8.0):
        t, y = ode.integrate(ode.rk4_step, osc.rhs, [0.0, v0], 0.0, 10.0, 1000)
        periods[str(v0)] = ode.zero_crossing_period(t, y[:, 0])
    out["rk4_period_by_initial_velocity"] = periods
    return out


# --------------------------------------------------------------------------- heat
def run_heat():
    x = np.linspace(0, 1, 101)
    start = np.full(101, 100.0)
    out = {}
    runs = {}
    for mat, dt, steps in [(heat.ALUMINIUM, 0.45, 10_000), (heat.IRON, 1.8, 2_500)]:
        times, snaps = heat.solve_ftcs(start, mat.alpha, 0.01, dt, steps, store_every=steps // 100)
        runs[mat.name] = (mat, times, snaps)
        out[mat.name] = {
            "alpha_m2_s": mat.alpha,
            "r": heat.stability_number(mat.alpha, 0.01, dt),
            "centre_temperature_t4500": float(snaps[-1][50]),
        }

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    picks = [5, 20, 40, 70, 100]
    for ax, (name, (mat, times, snaps)) in zip(axes, runs.items()):
        for shade, i in zip(np.linspace(0.4, 1.0, len(picks)), picks):
            ax.plot(x, snaps[i], color=SEQUENTIAL(shade), label=f"t = {times[i]:.0f} s")
        ax.set(title=name.capitalize(), xlabel="position (m)")
    axes[0].set_ylabel("temperature")
    axes[1].legend()
    save(fig, "heat_aluminium_vs_iron")

    mat, times, snaps = runs["aluminium"]
    out["aluminium"]["max_error_vs_series_t4500"] = float(
        np.max(np.abs(snaps[-1] - heat.analytic_uniform(x, times[-1], 1.0, mat.alpha)))
    )

    # convergence on the single-mode solution
    fig, ax = plt.subplots(figsize=(6, 4))
    out["spatial_order"] = {}
    for label, r, color in [("r = 0.4", 0.4, BLUE), ("r = 1/6", 1 / 6, ORANGE)]:
        pairs = [heat.sine_mode_error(n, r, t_end=500.0) for n in (11, 21, 41, 81, 161)]
        dx, err = map(np.array, zip(*pairs))
        slope = float(np.polyfit(np.log(dx), np.log(err), 1)[0])
        out["spatial_order"][label] = slope
        ax.loglog(dx, err, "o-", color=color, label=f"{label} (slope {slope:.2f})")
    ax.set(xlabel="grid spacing dx (m)", ylabel="max error at t = 500 s", title="Sine mode: error against exact solution")
    ax.legend()
    save(fig, "heat_convergence")

    # stability limit
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    alpha = heat.ALUMINIUM.alpha
    out["stability"] = {}
    for ax, r in zip(axes, (0.49, 0.51)):
        _, s = heat.solve_ftcs(start, alpha, 0.01, r * 0.01**2 / alpha, 400, store_every=400)
        out["stability"][f"max_abs_T_after_400_steps_r{r}"] = float(np.max(np.abs(s[-1])))
        ax.plot(x, s[-1], color=BLUE)
        ax.set(title=f"r = {r}, after 400 steps", xlabel="position (m)")
    axes[0].set_ylabel("temperature")
    save(fig, "heat_stability")

    # Newton cooling
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for label, h, color in [("insulated (h = 0)", 0.0, BLUE), ("cooling (h = 0.001)", 1e-3, ORANGE)]:
        t, s = heat.solve_ftcs(start, alpha, 0.01, 0.45, 10_000, h=h, store_every=50)
        out.setdefault("newton_cooling_centre_t4500", {})[label] = float(s[-1][50])
        ax.plot(t, s[:, 50], color=color, label=label)
    ax.set(xlabel="time (s)", ylabel="temperature at the centre", title="Extra losses to the environment")
    ax.legend()
    save(fig, "heat_newton_cooling")

    # two bars in contact
    xb = np.linspace(0, 0.5, 101)
    dx = xb[1] - xb[0]
    T0 = np.where(xb <= 0.25, 100.0, 50.0)
    t, s = heat.solve_ftcs(T0, alpha, dx, 0.1, 6000, store_every=20)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    mesh = ax.pcolormesh(xb, t, s, cmap="Blues", shading="auto")
    contours = ax.contour(xb, t, s, levels=[10, 20, 30, 40, 50, 60, 70], colors=INK, linewidths=0.8)
    ax.clabel(contours, fmt="%d", fontsize=8)
    ax.grid(False)
    fig.colorbar(mesh, ax=ax, label="temperature")
    ax.set(xlabel="position (m)", ylabel="time (s)", title="Two bars in contact (100 and 50)")
    save(fig, "heat_two_bars")
    out["two_bars_r"] = heat.stability_number(alpha, dx, 0.1)
    return out


# --------------------------------------------------------------------------- random numbers
def run_rng():
    n = 100_000
    generators = {
        "toy LCG (a=57, c=1, m=256)": rng.TOY.uniforms(n),
        "48-bit LCG (drand48 constants)": rng.GOOD.uniforms(n),
        "NumPy PCG64 (seed 0)": np.random.default_rng(0).random(n),
    }
    out = {"n_samples": n, "toy_period": rng.TOY.period(), "pair_chi_square_dof": 255, "generators": {}}
    for name, u in generators.items():
        out["generators"][name] = {
            "max_moment_error_k1_to_5": max(abs(v - 1 / (k + 1)) for k, v in rng.moments(u).items()),
            "lag1_correlation": rng.lag_correlation(u, 1),
            "pair_chi_square_16x16": rng.pair_chi_square(u),
        }

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    for ax, (name, u) in zip(axes, generators.items()):
        ax.scatter(u[:3000], u[1:3001], s=3, color=BLUE, linewidths=0)
        ax.set(title=name.split(" (")[0], xlabel="u(i)", aspect="equal", xlim=(0, 1), ylim=(0, 1))
    axes[0].set_ylabel("u(i+1)")
    save(fig, "rng_successive_pairs")

    ints = rng.RANDU.integers(6000)
    planes = rng.randu_plane_index(ints)
    out["randu_distinct_planes"] = int(len(np.unique(planes)))
    u = ints.astype(float) / rng.RANDU.m
    # look along a direction that lies inside the planes, so they show up edge-on
    normal = np.array([9.0, -6.0, 1.0])
    normal /= np.linalg.norm(normal)
    inside = np.cross(normal, [0.0, 0.0, 1.0])
    inside /= np.linalg.norm(inside)
    across = np.cross(inside, normal)
    pts = np.stack([u[:-2], u[1:-1], u[2:]], axis=1)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(pts @ normal, pts @ across, s=2, color=BLUE, linewidths=0)
    ax.set(
        xlabel="projection on the plane normal (9, -6, 1)",
        ylabel="projection inside the planes",
        title="RANDU triples, seen edge-on",
    )
    save(fig, "rng_randu_planes")
    return out


# --------------------------------------------------------------------------- random walk
def run_walk():
    gen = np.random.default_rng(2026)
    out = {}

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for color in (BLUE, ORANGE, AQUA):
        path = random_walk.walk(2000, 2, gen)
        ax.plot(*path.T, color=color, linewidth=0.9)
        ax.plot(*path[-1], "o", color=color, markersize=8, markeredgecolor="white", markeredgewidth=2)
    ax.plot(0, 0, "s", color=INK, markersize=8)
    ax.set(title="Three walks of 2,000 unit steps (square marks the start)", xlabel="x", ylabel="y", aspect="equal")
    save(fig, "walk_paths")

    n_values = np.array([10, 30, 100, 300, 1000, 3000, 10_000])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(n_values, np.sqrt(n_values), color=MUTED, linestyle="--", linewidth=1, label="sqrt(N)")
    out["n_walks_per_point"] = 2000
    out["scaling_exponent"] = {}
    for sampler, color, marker in [("uniform", BLUE, "o"), ("cube", ORANGE, "s")]:
        for dim in (2, 3):
            rms = [
                np.sqrt(random_walk.squared_distances(int(n), 2000, dim, gen, sampler).mean())
                for n in n_values
            ]
            out["scaling_exponent"][f"{sampler}_{dim}d"] = random_walk.scaling_exponent(n_values, rms)
            if dim == 2:
                ax.loglog(n_values, rms, marker, color=color, markersize=8, fillstyle="full" if sampler == "uniform" else "none", label=f"2D, {sampler} sampler")
    ax.set(xlabel="number of steps N", ylabel="root mean square distance", title="Diffusive scaling (2,000 walks per point)")
    ax.legend()
    save(fig, "walk_scaling")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    edges = np.linspace(-np.pi, np.pi, 73)
    for sampler, color in [("uniform", BLUE), ("cube", ORANGE)]:
        s = random_walk.unit_steps(1_000_000, 2, gen, sampler)
        ax.hist(np.arctan2(s[:, 1], s[:, 0]), bins=edges, density=True, histtype="step", linewidth=2, color=color, label=f"{sampler} sampler")
    theta = np.linspace(-np.pi, np.pi, 1000)
    ax.plot(theta, random_walk.cube_angle_density(theta), color=INK, linewidth=1, linestyle="--", label="exact cube density")
    ax.set(xlabel="step angle (rad)", ylabel="density", title="Normalising a point from a square favours the diagonals", ylim=(0, 0.33))
    ax.legend(ncol=3, loc="upper center")
    save(fig, "walk_step_angles")
    out["cube_density_diagonal_over_axis"] = float(
        random_walk.cube_angle_density(np.pi / 4) / random_walk.cube_angle_density(0.0)
    )
    return out


# --------------------------------------------------------------------------- decay
def run_decay():
    gen = np.random.default_rng(68111)
    out = {"p": 0.001, "n_steps": 1200, "fits": {}}
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharex=True, sharey=True)
    t = np.arange(1201)
    for ax, n0 in zip(axes, (10, 100, 1000, 10_000)):
        n = decay.simulate(n0, 0.001, 1200, gen)
        out["fits"][str(n0)] = decay.fit_decay_rate(n)
        ax.plot(t, n / n0, color=BLUE, label="simulation")
        ax.plot(t, np.exp(-decay.continuous_rate(0.001) * t), color=INK, linestyle="--", linewidth=1, label="exponential law")
        ax.set(title=f"N0 = {n0:,}", xlabel="time step", yscale="log", ylim=(0.2, 1.05))
        ax.yaxis.set_major_locator(FixedLocator([0.2, 0.3, 0.5, 0.7, 1.0]))
        ax.yaxis.set_major_formatter(FixedFormatter(["0.2", "0.3", "0.5", "0.7", "1.0"]))
        ax.yaxis.set_minor_locator(NullLocator())
    axes[0].set_ylabel("surviving fraction")
    axes[0].legend(fontsize=8)
    save(fig, "decay_population")

    # how good is the estimate, and what does it estimate?
    ps = np.array([0.001, 0.01, 0.03, 0.1, 0.2, 0.3])
    ratios = []
    for p in ps:
        steps = int(np.ceil(5 / p))
        n = decay.simulate(10_000_000, p, steps, gen)
        ratios.append(decay.fit_decay_rate(n, min_count=1000) / p)
    out["fitted_rate_over_p"] = {str(p): float(r) for p, r in zip(ps, ratios)}
    fig, ax = plt.subplots(figsize=(6, 3.8))
    grid = np.linspace(0.001, 0.31, 200)
    ax.plot(grid, -np.log1p(-grid) / grid, color=INK, linestyle="--", linewidth=1, label="-ln(1 - p) / p")
    ax.plot(ps, ratios, "o", color=BLUE, markersize=8, label="fitted slope / p")
    ax.axhline(1, color=MUTED, linewidth=1)
    ax.set(xlabel="decay probability per step p", ylabel="ratio", title="The log-linear fit estimates -ln(1 - p), not p")
    ax.legend()
    save(fig, "decay_rate_bias")

    spread = {}
    for n0 in (100, 1000, 10_000):
        fits = [decay.fit_decay_rate(decay.simulate(n0, 0.001, 1200, gen)) for _ in range(200)]
        spread[str(n0)] = {"mean": float(np.mean(fits)), "std": float(np.std(fits))}
    out["rate_estimate_over_200_runs"] = spread
    return out


SECTIONS = {"ode": run_ode, "heat": run_heat, "rng": run_rng, "walk": run_walk, "decay": run_decay}


def main(names):
    unknown = set(names) - set(SECTIONS)
    if unknown:
        raise SystemExit(f"unknown sections {sorted(unknown)}; choose from {list(SECTIONS)}")
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    summary_path = RESULTS / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for name in names or SECTIONS:
        print(f"running {name} ...", flush=True)
        summary[name] = SECTIONS[name]()
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
