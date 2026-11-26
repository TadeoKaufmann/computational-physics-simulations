import numpy as np
import pytest

from simphys import decay, random_walk, rng


def test_toy_generator_has_full_but_tiny_period():
    assert rng.TOY.period() == 256
    assert sorted(rng.TOY.integers(256).tolist()) == list(range(256))


def test_good_generator_passes_simple_checks():
    u = rng.GOOD.uniforms(100_000)
    assert u.min() >= 0 and u.max() < 1
    for k, value in rng.moments(u).items():
        assert value == pytest.approx(1 / (k + 1), abs=0.005)
    assert rng.lag_correlation(u, 1) == pytest.approx(0.25, abs=0.005)
    assert rng.pair_chi_square(u) < 400  # 255 degrees of freedom


def test_toy_generator_fails_the_pair_test():
    assert rng.pair_chi_square(rng.TOY.uniforms(100_000)) > 1000


def test_randu_triples_lie_on_few_planes():
    planes = rng.randu_plane_index(rng.RANDU.integers(10_000))
    assert len(np.unique(planes)) <= 15


@pytest.mark.parametrize("sampler", ["uniform", "cube"])
@pytest.mark.parametrize("dim", [2, 3])
def test_mean_square_distance_equals_number_of_steps(sampler, dim):
    r2 = random_walk.squared_distances(200, 4000, dim, np.random.default_rng(0), sampler)
    assert r2.mean() == pytest.approx(200, rel=0.08)


def test_steps_have_unit_length():
    steps = random_walk.unit_steps(100, 3, np.random.default_rng(0), "cube")
    np.testing.assert_allclose(np.linalg.norm(steps, axis=1), 1.0)


def test_cube_sampler_angle_density():
    s = random_walk.unit_steps(400_000, 2, np.random.default_rng(1), "cube")
    theta = np.arctan2(s[:, 1], s[:, 0])
    hist, edges = np.histogram(theta, bins=72, range=(-np.pi, np.pi), density=True)
    centres = (edges[:-1] + edges[1:]) / 2
    assert np.max(np.abs(hist - random_walk.cube_angle_density(centres))) < 0.01
    grid = np.linspace(-np.pi, np.pi, 100_001)
    assert np.trapz(random_walk.cube_angle_density(grid), grid) == pytest.approx(1.0, abs=1e-6)


def test_decay_population_never_increases():
    n = decay.simulate(1000, 0.05, 200, np.random.default_rng(0))
    assert n[0] == 1000 and np.all(np.diff(n) <= 0) and n[-1] >= 0


def test_fitted_rate_recovers_the_continuous_rate():
    p = 0.1
    n = decay.simulate(10_000_000, p, 60, np.random.default_rng(0))
    estimate = decay.fit_decay_rate(n, min_count=1000)
    assert estimate == pytest.approx(decay.continuous_rate(p), rel=0.01)
    assert abs(estimate - p) > 0.003
