import numpy as np
import pytest

from driftfdr.calibration import CalibrationConfig, NullDistribution, calibrate, fit_tail
from driftfdr.detectors import PageHinkley
from driftfdr.monitor import MonitorConfig, run_monitor
from driftfdr.streams import ScenarioConfig, make_scenario


def test_empirical_pvalue_formula():
    null = NullDistribution.from_samples(np.arange(99.0), 1, CalibrationConfig(tail="none"))
    assert null.pvalue(98.0)[0] == pytest.approx(2 / 100)
    assert null.pvalue(-1.0)[0] == pytest.approx(1.0)
    assert null.pvalue(1e9)[0] == pytest.approx(1 / 100)


def test_tail_extrapolates_below_resolution_and_is_monotone():
    samples = np.random.default_rng(0).exponential(2.0, size=2000)
    null = NullDistribution.from_samples(samples, 1, CalibrationConfig())
    assert null.has_tail
    grid = np.linspace(5, 40, 50)
    p = null.pvalue(grid)
    assert np.all(np.diff(p) <= 1e-12)
    assert p[-1] < 1 / 2001
    # exponential tail: P(X > 20) = exp(-10); the scale is estimated from 200 exceedances
    assert np.exp(-10) / 3 < null.pvalue(20.0)[0] < np.exp(-10) * 3


@pytest.mark.parametrize("model", ["exponential", "gpd"])
def test_tail_fit_on_exponential_samples(model):
    samples = np.sort(np.random.default_rng(1).exponential(3.0, size=5000))
    u, scale, shape, prob = fit_tail(samples, 0.1, model)
    assert shape < 0.15
    assert scale == pytest.approx(3.0, rel=0.2)
    assert prob == pytest.approx(0.1)


def test_calibrate_returns_one_null_per_horizon():
    x = np.random.default_rng(2).normal(size=300)
    nulls = calibrate(PageHinkley(), x, 100, horizon=3, config=CalibrationConfig(n_boot=100))
    assert len(nulls) == 3
    assert all(n.samples.size == 100 and np.all(np.diff(n.samples) >= 0) for n in nulls)
    # Page-Hinkley accumulates evidence, so later windows have larger null statistics
    assert np.median(nulls[2].samples) > np.median(nulls[0].samples)


@pytest.mark.parametrize("method", ["moving", "sieve"])
def test_pvalues_roughly_uniform_on_stationary_streams(method):
    sc = make_scenario(
        ScenarioConfig(n_streams=300, n_steps=800, phi=0.0, rho=0.0, drift_fraction=0.0), seed=3
    )
    cfg = MonitorConfig(horizon=3, calibration=CalibrationConfig(n_boot=200, method=method))
    p = run_monitor(sc, PageHinkley(), None, cfg).tests["pvalue"].to_numpy()
    assert p.min() > 0 and p.max() <= 1
    assert 0.02 < np.mean(p <= 0.05) < 0.09
    assert 0.40 < np.mean(p <= 0.5) < 0.60
