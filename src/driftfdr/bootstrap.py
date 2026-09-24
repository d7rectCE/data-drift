"""Resampling schemes for autocorrelated series.

* ``moving``: moving block bootstrap (Künsch, 1989), fixed block length;
* ``stationary``: stationary bootstrap (Politis & Romano, 1994), geometric
  block lengths with the given mean;
* ``sieve``: AR-sieve bootstrap (Bühlmann, 1997), an AR(p) fitted by
  Yule–Walker with AIC order selection, driven by resampled residuals; only
  meaningful for continuous signals;
* ``sieve_pu``: AR-sieve with parameter uncertainty. Each replicate refits the
  AR model on a series simulated from the fitted one and is generated from its
  own refitted coefficients, so the estimation error of the reference model
  (the source of the anti-conservative tails the plain sieve shows) is carried
  into the null distribution. Same idea as the correction of Wu & Apley for
  nested bootstraps.
* ``iid``: ordinary bootstrap, kept as a baseline that ignores dependence.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

METHODS = ("moving", "stationary", "sieve", "sieve_pu", "iid")


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


def _levinson_batch(acov: np.ndarray, order: int):
    """Yule–Walker coefficients of a fixed order for many autocovariance rows at once."""
    a = np.zeros((acov.shape[0], 0))
    err = acov[:, 0].copy()
    for k in range(1, order + 1):
        refl = (acov[:, k] - np.einsum("bi,bi->b", a, acov[:, k - 1 : 0 : -1])) / err
        a = np.concatenate([a - refl[:, None] * a[:, ::-1], refl[:, None]], axis=1)
        err = err * (1.0 - refl**2)
    return a, err


def ar_sieve_pu_series(x: np.ndarray, n_out: int, n_boot: int, rng, burn_in: int = 200):
    """AR-sieve replicates whose coefficients are redrawn per replicate; returns ``(series, order)``.

    Coefficients of replicate ``b`` are the Yule–Walker fit (same order as the
    original AIC choice) to a series of ``len(x)`` steps simulated from the
    model fitted to ``x``, i.e. a parametric bootstrap draw of the estimator.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    mu = x.mean()
    a = fit_ar_aic(x)
    p = a.size
    xc = x - mu
    if p == 0:
        # white noise: parameter uncertainty is only in the variance
        sims = rng.choice(xc, size=(n_boot, n))
        scale = sims.std(axis=1) / xc.std()
        return rng.choice(xc, size=(n_boot, n_out)) * scale[:, None] + mu, 0
    resid = xc[p:] - np.stack([xc[p - i : n - i] for i in range(1, p + 1)], axis=1) @ a
    resid = resid - resid.mean()
    ar = np.concatenate([[1.0], -a])
    sims = signal.lfilter([1.0], ar, rng.choice(resid, size=(n_boot, burn_in + n)), axis=1)[:, burn_in:]
    sims = sims - sims.mean(axis=1, keepdims=True)
    acov = np.stack([np.sum(sims[:, : n - k] * sims[:, k:], axis=1) / n for k in range(p + 1)], axis=1)
    a_b, err_b = _levinson_batch(acov, p)
    scale = np.sqrt(np.maximum(err_b, 1e-12) / np.mean(resid**2))
    innov = rng.choice(resid, size=(n_boot, burn_in + n_out)) * scale[:, None]
    y = np.zeros_like(innov)
    for t in range(innov.shape[1]):
        acc = innov[:, t].copy()
        for i in range(1, min(p, t) + 1):
            acc += a_b[:, i - 1] * y[:, t - i]
        y[:, t] = acc
    return y[:, burn_in:] + mu, p
