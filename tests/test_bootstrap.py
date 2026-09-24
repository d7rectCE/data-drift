import numpy as np
import pytest

from driftfdr.bootstrap import (
    METHODS,
    ar1_block_length,
    ar_sieve_series,
    bootstrap_indices,
    fit_ar_aic,
)
from driftfdr.streams import ar1_latent


@pytest.mark.parametrize("method", [m for m in METHODS if m != "sieve"])
def test_indices_shape_and_range(method):
    idx = bootstrap_indices(50, 130, 7, 6, method, np.random.default_rng(0))
    assert idx.shape == (7, 130)
    assert idx.min() >= 0 and idx.max() < 50


def test_moving_blocks_are_contiguous():
    idx = bootstrap_indices(100, 40, 3, 10, "moving", np.random.default_rng(1))
    blocks = idx.reshape(3, 4, 10)
    assert np.all(np.diff(blocks, axis=2) == 1)


def test_block_length_grows_with_autocorrelation():
    rng = np.random.default_rng(2)
    b_iid = ar1_block_length(ar1_latent(1, 400, 0.0, 0.0, rng)[0])
    b_ar = ar1_block_length(ar1_latent(1, 400, 0.8, 0.0, rng)[0])
    assert b_iid <= 3 < b_ar


def test_ar_fit_recovers_coefficient():
    x = ar1_latent(1, 5000, 0.6, 0.0, np.random.default_rng(3))[0]
    a = fit_ar_aic(x)
    assert a.size >= 1
    assert a[0] == pytest.approx(0.6, abs=0.05)


def test_sieve_preserves_autocorrelation():
    x = ar1_latent(1, 400, 0.7, 0.0, np.random.default_rng(4))[0]
    series, order = ar_sieve_series(x, 300, 200, np.random.default_rng(5))
    assert series.shape == (200, 300) and order >= 1
    xc = series - series.mean(axis=1, keepdims=True)
    r1 = np.mean(np.sum(xc[:, 1:] * xc[:, :-1], axis=1) / np.sum(xc * xc, axis=1))
    assert 0.55 < r1 < 0.8


def test_sieve_pu_spreads_coefficients_like_their_estimator():
    from driftfdr.bootstrap import ar_sieve_pu_series

    x = ar1_latent(1, 300, 0.5, 0.0, np.random.default_rng(6))[0]
    series, order = ar_sieve_pu_series(x, 1000, 400, np.random.default_rng(7))
    assert series.shape == (400, 1000) and order >= 1
    xc = series - series.mean(axis=1, keepdims=True)
    r1 = np.sum(xc[:, 1:] * xc[:, :-1], axis=1) / np.sum(xc * xc, axis=1)
    # replicates differ in their lag-1 autocorrelation by about the estimator's error sqrt((1 - phi^2) / n)
    assert 0.03 < r1.std() < 0.09
    plain, _ = ar_sieve_series(x, 1000, 400, np.random.default_rng(7))
    pc = plain - plain.mean(axis=1, keepdims=True)
    assert r1.std() > (np.sum(pc[:, 1:] * pc[:, :-1], axis=1) / np.sum(pc * pc, axis=1)).std()
