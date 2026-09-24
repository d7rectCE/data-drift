"""Synthetic multi-stream scenarios with known change points.

Each stream stands for the monitored signal of one production model, e.g. its
per-step loss. Streams are AR(1) in time and share a common factor, so both
serial and cross-stream dependence are present. A fraction of the streams
undergoes a single change in mean, abrupt or gradual, at a known time.
Changes are either sporadic (each stream at its own time) or clustered into
drift events that shift a whole group of streams at the same moment, as when
an upstream data source changes for many models at once.

Two views of the same latent process are exposed:

* ``values``: a continuous signal (loss proxy), used by Page-Hinkley, ADWIN, KS;
* ``errors``: binary error indicators ``1{latent > c}`` with base rate
  ``base_error_rate``, used by DDM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    fixed_onset: int | None = None
    """If set, every drifting stream changes at this step instead."""
    drift_events: int = 0
    """Number of drift events; each shifts ``event_fraction`` of the streams at once."""
    event_fraction: float = 0.1
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
    event: np.ndarray
    """Drift event of each stream, -1 for sporadic changes and stable streams."""
    later_changes: np.ndarray | None = field(default=None)
    """Optional ``(n_streams, m, 2)`` array of further (start, end) changes after the first."""

    def changes(self) -> tuple[np.ndarray, np.ndarray]:
        """All change starts and ends, shape ``(n_streams, n_changes)``, padded with ``NO_CHANGE``."""
        cs, ce = self.change_start[:, None], self.change_end[:, None]
        if self.later_changes is not None and self.later_changes.size:
            cs = np.concatenate([cs, self.later_changes[:, :, 0]], axis=1)
            ce = np.concatenate([ce, self.later_changes[:, :, 1]], axis=1)
        return cs, ce

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
        cs, ce = self.changes()
        start = np.asarray(start)[..., None] if np.ndim(start) else start
        stop = np.asarray(stop)[..., None] if np.ndim(stop) else stop
        return np.all((stop <= cs[streams]) | (start >= ce[streams]), axis=-1)


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

    event = np.full(n, -1, dtype=np.int64)
    lo, hi = int(config.onset_range[0] * T), int(config.onset_range[1] * T)

    def draw_onset():
        return config.fixed_onset if config.fixed_onset is not None else int(rng.integers(lo, hi))

    def apply_change(streams, tau):
        for k in streams:
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

    n_drift = int(round(config.drift_fraction * n))
    for k in rng.choice(n, size=n_drift, replace=False):
        apply_change([k], draw_onset())

    n_per_event = int(round(config.event_fraction * n))
    if config.drift_events * n_per_event > n - n_drift:
        raise ValueError("not enough stable streams for the requested drift events")
    for e in range(config.drift_events):
        stable = np.flatnonzero(change_start == NO_CHANGE)
        group = rng.choice(stable, size=n_per_event, replace=False)
        apply_change(group, draw_onset())
        event[group] = e

    latent = ar1_latent(n, T, config.phi, config.rho, rng)
    values = latent + shift
    errors = (values > stats.norm.isf(config.base_error_rate)).astype(np.int8)
    return Scenario(config, values, errors, change_start, change_end, kind, event)
