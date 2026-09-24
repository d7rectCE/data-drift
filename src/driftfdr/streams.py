"""Synthetic multi-stream scenarios with known change points.

Each stream stands for the monitored signal of one production model, e.g. its
per-step loss. Streams are AR(1) in time and share a common factor, so both
serial and cross-stream dependence are present. A fraction of the streams
undergoes a single change in mean, abrupt or gradual, at a known time.

Two views of the same latent process are exposed:

* ``values``: a continuous signal (loss proxy), used by Page-Hinkley, ADWIN, KS;
* ``errors``: binary error indicators ``1{latent > c}`` with base rate
  ``base_error_rate``, used by DDM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal, stats

NO_CHANGE = np.iinfo(np.int64).max // 2
DRIFT_TYPES = ("abrupt", "gradual", "mixed")


@dataclass(frozen=True)
class ScenarioConfig:
    n_streams: int = 100
    n_steps: int = 5000
    phi: float = 0.5
    """AR(1) coefficient of every stream."""
    rho: float = 0.3
    """Contemporaneous correlation between any two streams (common factor)."""
    drift_fraction: float = 0.1
    drift_type: str = "abrupt"
    magnitude: float = 1.0
    """Size of the mean shift, in units of the marginal standard deviation."""
    gradual_length: int = 300
    onset_range: tuple[float, float] = (0.3, 0.8)
    """Onsets are drawn uniformly from this fraction range of ``n_steps``."""
    base_error_rate: float = 0.2


@dataclass
class Scenario:
    config: ScenarioConfig
    values: np.ndarray
    errors: np.ndarray
    change_start: np.ndarray
    """First index at which the mean differs from its initial level."""
    change_end: np.ndarray
    """First index from which the mean stays at its final level."""
    drift_kind: np.ndarray

    @property
    def n_streams(self) -> int:
        return self.values.shape[0]

    @property
    def n_steps(self) -> int:
        return self.values.shape[1]

    @property
    def drifting(self) -> np.ndarray:
        return np.flatnonzero(self.change_start < NO_CHANGE)

    def signal(self, kind: str) -> np.ndarray:
        if kind == "values":
            return self.values
        if kind == "errors":
            return self.errors
        raise ValueError(f"unknown signal kind {kind!r}")

    def is_null(self, streams, start, stop) -> np.ndarray:
        """Whether the mean of each stream is constant on ``[start, stop)``.

        This is the regime-based null: a test comparing a reference segment with
        a later window is null iff no change happens anywhere between the start
        of the reference and the end of the window.
        """
        streams = np.asarray(streams)
        return (np.asarray(stop) <= self.change_start[streams]) | (
            np.asarray(start) >= self.change_end[streams]
        )


def ar1_latent(n_streams: int, n_steps: int, phi: float, rho: float, rng) -> np.ndarray:
    """Stationary unit-variance AR(1) streams with cross-correlation ``rho``."""
    common = rng.standard_normal(n_steps)
    idio = rng.standard_normal((n_streams, n_steps))
    innov = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idio
    if phi == 0.0:
        return innov
    eps = np.empty_like(innov)
    eps[:, 0] = innov[:, 0]
    eps[:, 1:], _ = signal.lfilter(
        [np.sqrt(1.0 - phi**2)], [1.0, -phi], innov[:, 1:], axis=1, zi=phi * innov[:, :1]
    )
    return eps


def make_scenario(config: ScenarioConfig, seed: int = 0) -> Scenario:
    if config.drift_type not in DRIFT_TYPES:
        raise ValueError(f"drift_type must be one of {DRIFT_TYPES}")
    rng = np.random.default_rng(seed)
    n, T = config.n_streams, config.n_steps
    shift = np.zeros((n, T))
    change_start = np.full(n, NO_CHANGE, dtype=np.int64)
    change_end = np.full(n, NO_CHANGE, dtype=np.int64)
    kind = np.full(n, "none", dtype=object)

    n_drift = int(round(config.drift_fraction * n))
    lo, hi = int(config.onset_range[0] * T), int(config.onset_range[1] * T)
    for k in rng.choice(n, size=n_drift, replace=False):
        tau = int(rng.integers(lo, hi))
        kind_k = config.drift_type
        if kind_k == "mixed":
            kind_k = str(rng.choice(["abrupt", "gradual"]))
        if kind_k == "abrupt":
            shift[k, tau:] = config.magnitude
            end = tau
        else:
            L = config.gradual_length
            ramp = np.minimum(1.0, (np.arange(tau, T) - tau + 1) / L)
            shift[k, tau:] = config.magnitude * ramp
            end = tau + L - 1
        change_start[k], change_end[k], kind[k] = tau, end, kind_k

    latent = ar1_latent(n, T, config.phi, config.rho, rng)
    values = latent + shift
    errors = (values > stats.norm.isf(config.base_error_rate)).astype(np.int8)
    return Scenario(config, values, errors, change_start, change_end, kind)
