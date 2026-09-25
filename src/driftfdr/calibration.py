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

import bisect
import math
from dataclasses import dataclass

import numpy as np

from .bootstrap import ar1_block_length, ar_sieve_pu_series, ar_sieve_series, bootstrap_indices
from .detectors import Detector

P_FLOOR = 1e-16


@dataclass(frozen=True)
class CalibrationConfig:
    """Defaults follow the experiments: AR-sieve with parameter uncertainty (moving
    blocks are used automatically for 0/1 signals), 2000 replicates and a GPD tail,
    which bring the per-window FWER of Bonferroni to its nominal level for
    Page-Hinkley and KS (experiment 10). Experiments 1-9 and 11-15 pin the earlier
    defaults (500 replicates, exponential tail) explicitly."""

    n_boot: int = 2000
    method: str = "sieve_pu"
    block_length: int | str = "auto"
    tail: str = "gpd"
    """Tail model for small p-values: ``exponential``, ``gpd`` or ``none``."""
    tail_fraction: float = 0.1
    min_exceedances: int = 10
    seed: int = 12345
    tolerance: float = 0.0
    """Null of *material* change: the new data may exceed the reference level by up to
    this much (signal units). The bootstrap continuation is shifted up by it, the least
    favourable point of that null, so p-values are valid for every smaller increase."""


@dataclass
class NullDistribution:
    """Bootstrap null distribution of a window statistic, with an optional fitted upper tail."""

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
        """Sort the bootstrap statistics and fit the tail model requested by ``config``."""
        samples = np.sort(np.asarray(samples, dtype=float))
        null = cls(samples, block_length, config.min_exceedances)
        if config.tail != "none":
            fit = fit_tail(samples, config.tail_fraction, config.tail)
            if fit is not None:
                null.tail_threshold, null.tail_scale, null.tail_shape, null.tail_prob = fit
        return null

    @property
    def has_tail(self) -> bool:
        """Whether a tail model was fitted (it is not for degenerate or tiny samples)."""
        return bool(np.isfinite(self.tail_scale))

    def pvalue(self, statistic) -> np.ndarray:
        """p-values of one or more observed statistics, as an array.

        ``(1 + #{T* >= T}) / (B + 1)``, replaced by the tail model when fewer than
        ``min_exceedances`` bootstrap statistics reach ``T``.
        """
        stat = np.atleast_1d(np.asarray(statistic, dtype=float))
        B = self.samples.size
        count = B - np.searchsorted(self.samples, stat, side="left")
        p = (1.0 + count) / (B + 1.0)
        if self.has_tail:
            use = (count < self.min_exceedances) & (stat > self.tail_threshold)
            if use.any():
                p[use] = self._tail_sf(stat[use])
        return p

    def pvalue_scalar(self, statistic: float) -> float:
        """``pvalue`` for a single statistic without numpy overhead (for per-step use)."""
        samples = self.__dict__.get("_sample_list")
        if samples is None:
            samples = self.__dict__["_sample_list"] = self.samples.tolist()
        B = len(samples)
        count = B - bisect.bisect_left(samples, statistic)
        if count < self.min_exceedances and math.isfinite(self.tail_scale) and statistic > self.tail_threshold:
            y = (statistic - self.tail_threshold) / self.tail_scale
            if self.tail_shape > 1e-12:
                base = 1.0 + self.tail_shape * y
                sf = base ** (-1.0 / self.tail_shape) if base > 0 else 0.0
            else:
                sf = math.exp(-y) if y < 700 else 0.0
            return min(1.0, max(P_FLOOR, self.tail_prob * sf))
        return (1.0 + count) / (B + 1.0)

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
    """Block length for a block method: 1 for iid, the AR(1) plug-in for ``"auto"``, else the given value."""
    if method == "iid":
        return 1
    if config.block_length == "auto":
        return ar1_block_length(reference, method)
    return int(config.block_length)


def bootstrap_series(reference, length: int, config: CalibrationConfig, rng, binary: bool = False):
    """``B`` pseudo series: a resampled reference followed by ``length`` resampled steps.

    With ``config.tolerance > 0`` the continuation is raised by the tolerance: added to
    continuous values, or, for binary errors, by turning zeros into ones with the
    probability that raises the error rate by that much.
    """
    series, param = _resample(reference, length, config, rng, binary)
    if config.tolerance > 0:
        n_ref = reference.size
        new = series[:, n_ref:]
        if binary:
            p0 = float(np.mean(reference))
            flip = rng.random(new.shape) < min(1.0, config.tolerance / max(1e-9, 1.0 - p0))
            series[:, n_ref:] = np.where(flip, 1.0, new)
        else:
            series[:, n_ref:] = new + config.tolerance
    return series, param


def _resample(reference, length: int, config: CalibrationConfig, rng, binary: bool):
    """Resampled reference plus continuation, before any tolerance shift.

    Reference and continuation are resampled independently, since the tested
    data are generally not adjacent to the reference. The sieve bootstrap
    produces continuous values, so binary signals fall back to moving blocks.
    Returns ``(series, param)`` where ``param`` is the block length, or the AR
    order for the sieve.
    """
    n_ref = reference.size
    B = config.n_boot
    method = "moving" if (config.method.startswith("sieve") and binary) else config.method
    if method == "sieve_pu":
        # one path with shared coefficients; the gap decorrelates reference and new data
        gap = 200
        path, order = ar_sieve_pu_series(reference, n_ref + gap + length, B, rng)
        return np.concatenate([path[:, :n_ref], path[:, n_ref + gap :]], axis=1), order
    if method == "sieve":
        ref_part, order = ar_sieve_series(reference, n_ref, B, rng)
        new_part, _ = ar_sieve_series(reference, length, B, rng)
        return np.concatenate([ref_part, new_part], axis=1), order
    b = resolve_block_length(reference, method, config)
    ref_idx = bootstrap_indices(n_ref, n_ref, B, b, method, rng)
    new_idx = bootstrap_indices(n_ref, length, B, b, method, rng)
    return reference[np.concatenate([ref_idx, new_idx], axis=1)].astype(float), b


def calibrate_many(
    detector: Detector,
    references: np.ndarray,
    window: int,
    horizon: int,
    config: CalibrationConfig,
    seeds,
    max_chunk_elements: int = 4_000_000,
) -> list[list[NullDistribution]]:
    """Bootstrap null distributions for several streams at once.

    ``references`` has shape ``(n_streams, n_ref)``. For every stream the
    result holds ``horizon`` null distributions: entry ``h - 1`` is for the
    statistic of the last window when the detector has seen ``h`` windows
    since the reference. Because scores are causal, all of them come from a
    single pass over bootstrap series of ``n_ref + horizon * window`` steps.
    ``seeds`` gives one seed (anything ``np.random.default_rng`` accepts) per
    stream, so results do not depend on which streams are calibrated together.
    """
    references = np.atleast_2d(np.asarray(references, dtype=float))
    n_streams, n_ref = references.shape
    length = horizon * window
    B = config.n_boot
    # binary by declaration (DDM) or by content (any detector fed 0/1 errors)
    binary = detector.input_kind == "errors" or bool(np.isin(references, (0.0, 1.0)).all())
    chunk = max(1, max_chunk_elements // (B * (n_ref + length)))
    out: list[list[NullDistribution]] = []
    for c0 in range(0, n_streams, chunk):
        rows = range(c0, min(n_streams, c0 + chunk))
        series, params = [], []
        for a in rows:
            rng = np.random.default_rng(seeds[a])
            s, param = bootstrap_series(references[a], length, config, rng, binary)
            series.append(s)
            params.append(param)
        stats = detector.window_statistics(np.concatenate(series), n_ref, window)
        stats = stats.reshape(len(rows), B, horizon)
        for a, st, b in zip(rows, stats, params):
            if _untestable(references[a], binary, config.tolerance):
                st = np.full_like(st, np.inf)  # every p-value becomes 1
            out.append([NullDistribution.from_samples(st[:, h], b, config) for h in range(horizon)])
    return out


def _untestable(reference: np.ndarray, binary: bool, tolerance: float) -> bool:
    """A constant reference carries no information on variability, and a binary error
    rate already within ``tolerance`` of 1 cannot rise materially; such streams get p = 1."""
    if np.ptp(reference) == 0:
        return True
    return binary and reference.mean() + tolerance >= 1.0


def calibrate(
    detector: Detector, reference, window: int, horizon: int = 1, config=CalibrationConfig(), seed=0
) -> list[NullDistribution]:
    """Null distributions for one stream, one per number of windows seen."""
    return calibrate_many(detector, np.asarray(reference)[None], window, horizon, config, [seed])[0]
