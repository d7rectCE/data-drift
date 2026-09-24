"""Multi-stream monitoring loop with retraining on alarm.

Time is split into check windows of ``window`` steps. Every stream owns a
reference segment of ``n_ref`` steps collected right after its last
(re)training. At the end of each window, every stream whose reference is
complete is tested: its detector is warmed up on the reference and run over
the most recent ``min(elapsed, horizon)`` windows, and the statistic is the
maximum score inside the current window. The look-back lets evidence
accumulate for up to ``horizon`` windows, as it does for a detector running
continuously in production, while keeping one calibration per reference.

The procedure then decides which streams alarm; alarmed streams are retrained
and their new reference is the next ``n_ref`` steps, during which they are
not monitored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .calibration import CalibrationConfig, NullDistribution, calibrate_many
from .detectors import Detector
from .online_fdr import Procedure
from .streams import Scenario


@dataclass(frozen=True)
class MonitorConfig:
    n_ref: int = 300
    window: int = 100
    horizon: int = 5
    """Look-back in windows; 1 compares each window with the reference alone."""
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    def __post_init__(self):
        if self.n_ref % self.window:
            raise ValueError("n_ref must be a multiple of window")


@dataclass
class MonitorResult:
    tests: pd.DataFrame
    """One row per (stream, window) test: statistic, p-value, decision, ground truth."""
    n_windows: int
    scenario: Scenario
    config: MonitorConfig


NullCache = dict[tuple[int, int], list[NullDistribution]]


def run_monitor(
    scenario: Scenario,
    detector: Detector,
    procedure: Procedure | None,
    config: MonitorConfig = MonitorConfig(),
    seed: int = 0,
    cache: NullCache | None = None,
) -> MonitorResult:
    """Run one monitoring pass.

    ``procedure=None`` never alarms and only records statistics and p-values,
    which is how null calibration is checked. ``cache`` maps
    ``(stream, reference_start)`` to calibrated null distributions; pass the
    same dict to runs on the same scenario, detector and config (e.g. different
    procedures) to reuse calibrations.
    """
    x = scenario.signal(detector.input_kind).astype(float)
    n_streams, n_steps = x.shape
    n_ref, W, H = config.n_ref, config.window, config.horizon
    cache = {} if cache is None else cache
    need_pvalues = procedure is None or not procedure.uses_statistics
    rng = np.random.default_rng(seed)
    ref_start = np.zeros(n_streams, dtype=np.int64)
    chunks = []
    window_starts = range(n_ref, n_steps - W + 1, W)
    for j, t0 in enumerate(window_starts):
        t1 = t0 + W
        active = np.flatnonzero(ref_start + n_ref <= t0)
        if active.size == 0:
            continue
        seen = np.minimum((t1 - ref_start[active] - n_ref) // W, H)
        stats = np.empty(active.size)
        for h in np.unique(seen):
            group = active[seen == h]
            series = np.concatenate([_references(x, group, ref_start, n_ref), x[group, t1 - h * W : t1]], axis=1)
            stats[seen == h] = detector.window_statistics(series, n_ref, W)[:, -1]
        pvals = np.full(active.size, np.nan)
        if need_pvalues:
            _fill_cache(cache, detector, x, active, ref_start, config)
            for i, (k, h) in enumerate(zip(active, seen)):
                pvals[i] = cache[(k, ref_start[k])][h - 1].pvalue(stats[i])[0]
        if procedure is None:
            rejected = np.zeros(active.size, dtype=bool)
        else:
            rejected = procedure.decide(stats if procedure.uses_statistics else pvals, rng)
        chunks.append(
            pd.DataFrame(
                {
                    "window": j,
                    "t_end": t1,
                    "stream": active,
                    "ref_start": ref_start[active],
                    "windows_seen": seen,
                    "statistic": stats,
                    "pvalue": pvals,
                    "rejected": rejected,
                    "is_null": scenario.is_null(active, ref_start[active], t1),
                }
            )
        )
        ref_start[active[rejected]] = t1
    return MonitorResult(pd.concat(chunks, ignore_index=True), len(window_starts), scenario, config)


def _references(x, streams, ref_start, n_ref):
    return np.stack([x[k, ref_start[k] : ref_start[k] + n_ref] for k in streams])


def _fill_cache(cache, detector, x, active, ref_start, config):
    missing = [k for k in active if (k, ref_start[k]) not in cache]
    if not missing:
        return
    refs = _references(x, missing, ref_start, config.n_ref)
    seeds = [[config.calibration.seed, int(k), int(ref_start[k])] for k in missing]
    nulls = calibrate_many(detector, refs, config.window, config.horizon, config.calibration, seeds)
    for k, null in zip(missing, nulls):
        cache[(k, ref_start[k])] = null
