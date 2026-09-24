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
    magnitude_range: tuple[float, float] | None = None
    """If set, each drifting stream draws its shift uniformly from this range instead."""
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
    truth: np.ndarray | None = field(default=None)
    """Optional ``(n_streams, n_steps)`` oracle level of the monitored quantity (e.g. the
    smoothed true error rate). When set, a test is null iff the level in the tested
    window exceeds the level over the reference by at most ``tolerance``: the null of
    *material degradation* instead of *any change*."""
    tolerance: float = 0.0
    truth_ref: np.ndarray | None = field(default=None, repr=False)
    """Optional level used for the reference side of the oracle comparison (defaults to ``truth``)."""
    mean_shift: np.ndarray | None = field(default=None, repr=False)
    """True mean shift of every stream (synthetic scenarios only)."""
    features: np.ndarray | None = field(default=None, repr=False)
    """Model input of every stream, for detectors that watch p(X) (supervised scenarios)."""

    def with_material_null(
        self, tolerance: float, truth: np.ndarray | None = None, truth_ref: np.ndarray | None = None
    ) -> "Scenario":
        """Copy whose ground truth is *material degradation*: level up by more than ``tolerance``.

        ``truth`` defaults to the true mean shift of a synthetic scenario.
        """
        from dataclasses import replace

        truth = self.mean_shift if truth is None else truth
        if truth is None:
            raise ValueError("no oracle level available for this scenario")
        ref = None if truth_ref is None else np.asarray(truth_ref, dtype=float)
        return replace(self, truth=np.asarray(truth, dtype=float), truth_ref=ref, tolerance=tolerance)

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
        if kind == "features":
            return self.features
        if kind == "errors":
            return self.errors
        raise ValueError(f"unknown signal kind {kind!r}")

    def _cumsum(self, name):
        cache = self.__dict__.setdefault("_cumsums", {})
        arr = getattr(self, name)
        if name not in cache or cache[name].shape[1] != arr.shape[1] + 1:
            cache[name] = np.concatenate([np.zeros((arr.shape[0], 1)), np.cumsum(arr, axis=1)], axis=1)
        return cache[name]

    def is_null(self, streams, start, stop, ref_len=None, window=None) -> np.ndarray:
        """Ground truth of the test comparing the reference at ``start`` with the window ending at ``stop``.

        Regime null (default): no change anywhere between the start of the
        reference and the end of the window. With ``truth`` set: the oracle level
        in the last ``window`` steps exceeds its mean over the ``ref_len``-step
        reference by at most ``tolerance``.
        """
        streams = np.asarray(streams)
        if self.truth is not None:
            if ref_len is None or window is None:
                raise ValueError("ref_len and window are needed with an oracle truth")
            c = self._cumsum("truth")
            cr = c if self.truth_ref is None else self._cumsum("truth_ref")
            start, stop = np.asarray(start), np.asarray(stop)
            ref = (cr[streams, start + ref_len] - cr[streams, start]) / ref_len
            cur = (c[streams, stop] - c[streams, stop - window]) / window
            return cur - ref <= self.tolerance
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
            size = config.magnitude
            if config.magnitude_range is not None:
                size = float(rng.uniform(*config.magnitude_range))
            kind_k = config.drift_type
            if kind_k == "mixed":
                kind_k = str(rng.choice(["abrupt", "gradual"]))
            if kind_k == "abrupt":
                shift[k, tau:] = size
                end = tau
            else:
                L = config.gradual_length
                ramp = np.minimum(1.0, (np.arange(tau, T) - tau + 1) / L)
                shift[k, tau:] = size * ramp
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
    return Scenario(config, values, errors, change_start, change_end, kind, event, mean_shift=shift)


SUPERVISED_KINDS = ("virtual", "real", "both", "cyclic")


@dataclass(frozen=True)
class SupervisedConfig:
    """Streams of a feature, a label and a fixed linear model ``y_hat = beta0 * x``.

    ``virtual`` drift shifts the mean of x (p(X) changes, the model stays right);
    ``real`` drift changes the slope (p(y|X) changes, the error rises);
    ``both`` does both at once; ``cyclic`` switches the slope back and forth every
    ``period`` steps after the onset (a recurring concept).
    """

    n_streams: int = 100
    n_steps: int = 5000
    phi: float = 0.5
    rho: float = 0.0
    drift_fraction: float = 0.6
    kinds: tuple[str, ...] = ("virtual", "real", "both")
    feature_shift: float = 1.0
    """Shift of the feature mean for virtual drift, in feature standard deviations."""
    slope_change: float = 0.5
    """Change of the slope for real drift (the model slope is 1, noise sd is 1)."""
    period: int = 1000
    onset_range: tuple[float, float] = (0.3, 0.6)
    error_quantile: float = 0.8
    """0/1 errors mark squared residuals above this quantile of the pre-drift residual."""


def make_supervised_scenario(config: SupervisedConfig, seed: int = 0) -> Scenario:
    """Scenario whose ``values`` are squared residuals of the model and ``features`` the input x.

    Ground truth: ``mean_shift`` holds the true expected loss increase, so
    ``with_material_null`` judges alarms by model degradation; change points mark
    every change of p(X) or p(y|X) for the regime null.
    """
    for k in config.kinds:
        if k not in SUPERVISED_KINDS:
            raise ValueError(f"unknown drift kind {k!r}")
    rng = np.random.default_rng(seed)
    n, T = config.n_streams, config.n_steps
    x = ar1_latent(n, T, config.phi, config.rho, rng)
    noise = ar1_latent(n, T, config.phi, 0.0, rng)
    x_shift = np.zeros((n, T))
    slope = np.ones((n, T))
    n_drift = int(round(config.drift_fraction * n))
    drifting = rng.choice(n, size=n_drift, replace=False)
    kind = np.full(n, "none", dtype=object)
    lo, hi = int(config.onset_range[0] * T), int(config.onset_range[1] * T)
    changes = [[] for _ in range(n)]
    for i, k in enumerate(drifting):
        kind_k = config.kinds[i % len(config.kinds)]
        tau = int(rng.integers(lo, hi))
        kind[k] = kind_k
        if kind_k in ("virtual", "both"):
            x_shift[k, tau:] = config.feature_shift
        if kind_k in ("real", "both"):
            slope[k, tau:] = 1.0 + config.slope_change
        if kind_k == "cyclic":
            on = ((np.arange(tau, T) - tau) // config.period) % 2 == 0
            slope[k, tau:] = np.where(on, 1.0 + config.slope_change, 1.0)
            changes[k] = list(range(tau, T, config.period))
        else:
            changes[k] = [tau]
    feat = x + x_shift
    y = slope * feat + noise
    resid = y - feat  # model slope is 1
    loss = resid**2
    # expected loss: E[(slope-1)^2 x^2] + 1, with E[x^2] = 1 + shift^2
    expected = (slope - 1.0) ** 2 * (1.0 + x_shift**2) + 1.0
    threshold = np.quantile(noise**2, config.error_quantile)
    errors = (loss > threshold).astype(np.int8)
    m = max(len(c) for c in changes) if n_drift else 1
    cs = np.full((n, m), NO_CHANGE, dtype=np.int64)
    for k, c in enumerate(changes):
        cs[k, : len(c)] = c
    later = np.stack([cs[:, 1:], cs[:, 1:]], axis=-1) if m > 1 else None
    sc = Scenario(
        ScenarioConfig(n_streams=n, n_steps=T, phi=config.phi, rho=config.rho, drift_fraction=config.drift_fraction),
        loss,
        errors,
        change_start=cs[:, 0].copy(),
        change_end=cs[:, 0].copy(),
        drift_kind=kind,
        event=np.full(n, -1, dtype=np.int64),
        later_changes=later,
        mean_shift=expected - 1.0,
    )
    sc.features = feat
    return sc
