"""Resampling schemes for autocorrelated series.

* ``moving``: moving block bootstrap (Künsch, 1989), fixed block length;
* ``stationary``: stationary bootstrap (Politis & Romano, 1994), geometric
  block lengths with the given mean;
* ``sieve``: AR-sieve bootstrap (Bühlmann, 1997), an AR(p) fitted by
  Yule–Walker with AIC order selection, driven by resampled residuals; only
  meaningful for continuous signals;
* ``iid``: ordinary bootstrap, kept as a baseline that ignores dependence.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

METHODS = ("moving", "stationary", "sieve", "iid")


def ar1_block_length(x: np.ndarray, method: str = "moving") -> int:
    """Plug-in block length for an AR(1) approximation of ``x``.

    Uses the Politis–White (2004) optimal-rate formula specialised to AR(1):
    ``b = c * (2 r / (1 - r^2))^(2/3) * n^(1/3)``, with ``c = (3/2)^(1/3)``
    for block bootstrap and ``c = 1`` for the stationary bootstrap, where ``r``
    is the lag-1 autocorrelation.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    xc = x - x.mean()
    denom = float(xc @ xc)
    if n < 8 or denom == 0.0:
        return 1
    r = float(np.clip((xc[1:] @ xc[:-1]) / denom, 0.0, 0.95))
    if r == 0.0:
        return 1
    c = 1.0 if method == "stationary" else 1.5 ** (1 / 3)
    b = c * (2 * r / (1 - r**2)) ** (2 / 3) * n ** (1 / 3)
    return int(np.clip(np.ceil(b), 1, max(1, n // 4)))


def bootstrap_indices(
    n_source: int, n_out: int, n_boot: int, block_length: int, method: str, rng
) -> np.ndarray:
    """Indices into a source series of length ``n_source``, shape ``(n_boot, n_out)``."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if method == "iid" or block_length <= 1:
        return rng.integers(0, n_source, size=(n_boot, n_out))
    if method == "moving":
        b = min(block_length, n_source)
        n_blocks = -(-n_out // b)
        starts = rng.integers(0, n_source - b + 1, size=(n_boot, n_blocks))
        idx = starts[:, :, None] + np.arange(b)
        return idx.reshape(n_boot, n_blocks * b)[:, :n_out]
    # stationary bootstrap: start a new block with probability 1/b, wrap around
    new_block = rng.random((n_boot, n_out)) < 1.0 / block_length
    new_block[:, 0] = True
    starts = rng.integers(0, n_source, size=(n_boot, n_out))
    pos = np.arange(n_out)
    last = np.maximum.accumulate(np.where(new_block, pos, 0), axis=1)
    return (np.take_along_axis(starts, last, axis=1) + pos - last) % n_source


def fit_ar_aic(x: np.ndarray, max_order: int | None = None) -> np.ndarray:
    """Yule–Walker AR coefficients ``a`` (``x_t = sum_i a_i x_{t-i} + e_t``), order by AIC."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = x.size
    if max_order is None:
        max_order = int(min(10 * np.log10(n), n // 10))
    acov = np.array([x[: n - k] @ x[k:] / n for k in range(max_order + 1)])
    if acov[0] <= 0:
        return np.zeros(0)
    # Levinson–Durbin recursion gives every order up to max_order at once
    best_aic, best = n * np.log(acov[0]), np.zeros(0)
    a, err = np.zeros(0), acov[0]
    for p in range(1, max_order + 1):
        k = (acov[p] - a @ acov[p - 1 : 0 : -1]) / err
        a = np.concatenate([a - k * a[::-1], [k]])
        err *= 1.0 - k * k
        if err <= 0:
            break
        aic = n * np.log(err) + 2 * p
        if aic < best_aic:
            best_aic, best = aic, a.copy()
    return best


def ar_sieve_series(x: np.ndarray, n_out: int, n_boot: int, rng, burn_in: int = 200):
    """AR-sieve bootstrap replicates of ``x``; returns ``(series, order)``."""
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    a = fit_ar_aic(x)
    p = a.size
    xc = x - mu
    if p:
        resid = xc[p:] - np.stack([xc[p - i : x.size - i] for i in range(1, p + 1)], axis=1) @ a
    else:
        resid = xc
    resid = resid - resid.mean()
    innov = rng.choice(resid, size=(n_boot, burn_in + n_out))
    series = signal.lfilter([1.0], np.concatenate([[1.0], -a]), innov, axis=1)
    return series[:, burn_in:] + mu, p
