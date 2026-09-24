"""Turning detector statistics into p-values by resampling.

For a stream with reference segment ``R`` the null distribution of the window
statistic is estimated by scoring ``B`` bootstrap series resampled from ``R``:
a pseudo-reference of ``len(R)`` steps followed by a pseudo-window. Both parts
are resampled, so their mutual variability matches that of a fresh window
compared with the observed reference. The p-value is the usual
``(1 + #{T* >= T}) / (B + 1)``.

With ``B`` in the hundreds the smallest attainable p-value is ``1 / (B + 1)``,
far above the per-test levels online FDR procedures reach with hundreds of
streams. When fewer than ``min_exceedances`` bootstrap statistics exceed the
observed one, the p-value is extrapolated from a tail model fitted to the top
``tail_fraction`` of the bootstrap sample (Knijnenburg et al., 2009). The
default tail is exponential: the statistics here are maxima of sums or
squared standardised differences, which lie in the Gumbel domain, and fixing
the shape avoids the large variance of an estimated GPD shape. A GPD fit by
probability-weighted moments is available as ``tail="gpd"``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bootstrap import ar1_block_length, ar_sieve_series, bootstrap_indices
from .detectors import Detector

P_FLOOR = 1e-16


@dataclass(frozen=True)
class CalibrationConfig:
    n_boot: int = 500
    method: str = "moving"
    block_length: int | str = "auto"
    tail: str = "exponential"
    """Tail model for small p-values: ``exponential``, ``gpd`` or ``none``."""
    tail_fraction: float = 0.1
    min_exceedances: int = 10
    seed: int = 12345


@dataclass
class NullDistribution:
    samples: np.ndarray
    """Sorted bootstrap statistics (may contain ``inf``)."""
    block_length: int
    """Block length, or the AR order for the sieve bootstrap."""
    min_exceedances: int = 10
    tail_threshold: float = np.nan
    tail_scale: float = np.nan
    tail_shape: float = np.nan
    tail_prob: float = np.nan

    @classmethod
    def from_samples(
        cls, samples: np.ndarray, block_length: int, config: CalibrationConfig
    ) -> NullDistribution:
        samples = np.sort(np.asarray(samples, dtype=float))
        null = cls(samples, block_length, config.min_exceedances)
        if config.tail != "none":
            fit = fit_tail(samples, config.tail_fraction, config.tail)
            if fit is not None:
                null.tail_threshold, null.tail_scale, null.tail_shape, null.tail_prob = fit
        return null

    @property
    def has_tail(self) -> bool:
        return bool(np.isfinite(self.tail_scale))

    def pvalue(self, statistic) -> np.ndarray:
        stat = np.atleast_1d(np.asarray(statistic, dtype=float))
        B = self.samples.size
        count = B - np.searchsorted(self.samples, stat, side="left")
        p = (1.0 + count) / (B + 1.0)
        if self.has_tail:
            use = (count < self.min_exceedances) & (stat > self.tail_threshold)
            if use.any():
                p[use] = self._tail_sf(stat[use])
        return p

    def _tail_sf(self, stat: np.ndarray) -> np.ndarray:
        y = (stat - self.tail_threshold) / self.tail_scale
        with np.errstate(over="ignore", invalid="ignore"):
            if self.tail_shape > 1e-12:
                sf = (1.0 + self.tail_shape * y) ** (-1.0 / self.tail_shape)
            else:
                sf = np.exp(-y)
        sf = np.where(np.isfinite(sf), sf, 0.0)
        return np.clip(self.tail_prob * sf, P_FLOOR, 1.0)


def fit_tail(
    sorted_samples: np.ndarray, tail_fraction: float, model: str = "exponential", min_tail: int = 30
):
    """Fit an exponential or GPD tail to the largest bootstrap statistics.

    The GPD is fitted by probability-weighted moments (Hosking & Wallis, 1987)
    with the shape clipped to ``[0, 0.9]``. Returns
    ``(threshold, scale, shape, P(T > threshold))`` or ``None``.
    """
    if model not in ("exponential", "gpd"):
        raise ValueError("tail model must be 'exponential', 'gpd' or 'none'")
    B = sorted_samples.size
    finite = sorted_samples[np.isfinite(sorted_samples)]
    n_exc = max(min_tail, int(tail_fraction * B))
    if finite.size < n_exc + 1:
        return None
    u = finite[-(n_exc + 1)]
    y = finite[-n_exc:] - u
    if y.max() <= 0:
        return None
    prob = (n_exc + B - finite.size) / B
    shape, scale = 0.0, y.mean()
    if model == "gpd":
        a0 = y.mean()
        a1 = np.mean((1.0 - (np.arange(1, n_exc + 1) - 0.35) / n_exc) * y)
        denom = a0 - 2.0 * a1
        if denom > 0:
            xi, sigma = 2.0 - a0 / denom, 2.0 * a0 * a1 / denom
            if xi > 0 and sigma > 0:
                shape, scale = min(xi, 0.9), sigma
    return float(u), float(scale), float(shape), prob


def resolve_block_length(reference: np.ndarray, method: str, config: CalibrationConfig) -> int:
    if method == "iid":
        return 1
    if config.block_length == "auto":
        return ar1_block_length(reference, method)
    return int(config.block_length)


def bootstrap_series(reference, window: int, config: CalibrationConfig, rng, binary: bool = False):
    """``B`` pseudo "reference + window" series resampled from ``reference``.

    Reference and window parts are resampled independently, since the tested
    window is generally not adjacent to the reference. The sieve bootstrap
    produces continuous values, so binary signals fall back to moving blocks.
    Returns ``(series, param)`` where ``param`` is the block length, or the AR
    order for the sieve.
    """
    n_ref = reference.size
    B = config.n_boot
    method = "moving" if (config.method == "sieve" and binary) else config.method
    if method == "sieve":
        ref_part, order = ar_sieve_series(reference, n_ref, B, rng)
        win_part, _ = ar_sieve_series(reference, window, B, rng)
        return np.concatenate([ref_part, win_part], axis=1), order
    b = resolve_block_length(reference, method, config)
    ref_idx = bootstrap_indices(n_ref, n_ref, B, b, method, rng)
    win_idx = bootstrap_indices(n_ref, window, B, b, method, rng)
    return reference[np.concatenate([ref_idx, win_idx], axis=1)], b


def calibrate_many(
    detector: Detector,
    references: np.ndarray,
    window: int,
    config: CalibrationConfig,
    seeds,
    max_chunk_elements: int = 4_000_000,
) -> list[NullDistribution]:
    """Bootstrap null distributions for several streams at once.

    ``references`` has shape ``(n_streams, n_ref)``; ``seeds`` gives one seed
    (anything ``np.random.default_rng`` accepts) per stream so that results do
    not depend on which streams are calibrated together.
    """
    references = np.atleast_2d(np.asarray(references, dtype=float))
    n_streams, n_ref = references.shape
    n_out = n_ref + window
    B = config.n_boot
    binary = detector.input_kind == "errors"
    chunk = max(1, max_chunk_elements // (B * n_out))
    out: list[NullDistribution] = []
    for c0 in range(0, n_streams, chunk):
        rows = range(c0, min(n_streams, c0 + chunk))
        series, params = [], []
        for a in rows:
            s, param = bootstrap_series(
                references[a], window, config, np.random.default_rng(seeds[a]), binary
            )
            series.append(s)
            params.append(param)
        stats = detector.window_statistic(np.concatenate(series), n_ref).reshape(len(rows), B)
        out.extend(NullDistribution.from_samples(s, b, config) for s, b in zip(stats, params))
    return out


def calibrate(detector: Detector, reference, window: int, config=CalibrationConfig(), seed=0):
    return calibrate_many(detector, np.asarray(reference)[None], window, config, [seed])[0]
