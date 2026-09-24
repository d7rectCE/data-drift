"""Multi-stream monitoring loop with retraining on alarm.

Time is split into check windows of ``window`` steps. Every stream owns a
reference segment of ``n_ref`` steps collected right after its last
(re)training. At the end of each window, every stream whose reference is
complete is tested ("window vs reference"), the procedure decides which
streams alarm, and alarmed streams are retrained: their new reference is the
next ``n_ref`` steps, during which they are not monitored.
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
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)


@dataclass
class MonitorResult:
    tests: pd.DataFrame
    """One row per (stream, window) test: p-value, decision, ground truth."""
    n_windows: int
    scenario: Scenario
    config: MonitorConfig


NullCache = dict[tuple[int, int], NullDistribution]


def run_monitor(
    scenario: Scenario,
    detector: Detector,
    procedure: Procedure,
    config: MonitorConfig = MonitorConfig(),
    seed: int = 0,
    cache: NullCache | None = None,
) -> MonitorResult:
    """Run one monitoring pass.

    ``cache`` maps ``(stream, reference_start)`` to a calibrated null
    distribution. Pass the same dict to runs on the same scenario, detector and
    config (e.g. different procedures) to reuse calibrations.
    """
    x = scenario.signal(detector.input_kind).astype(float)
    n_streams, n_steps = x.shape
    n_ref, W = config.n_ref, config.window
    cache = {} if cache is None else cache
    rng = np.random.default_rng(seed)
    ref_start = np.zeros(n_streams, dtype=np.int64)
    chunks = []
    window_starts = range(n_ref, n_steps - W + 1, W)
    for j, t0 in enumerate(window_starts):
        t1 = t0 + W
        active = np.flatnonzero(ref_start + n_ref <= t0)
        if active.size == 0:
            continue
        refs = np.stack([x[k, ref_start[k] : ref_start[k] + n_ref] for k in active])
        stats = detector.window_statistic(np.concatenate([refs, x[active, t0:t1]], axis=1), n_ref)
        if procedure.uses_statistics:
            pvals = np.full(active.size, np.nan)
            rejected = procedure.decide(stats, rng)
        else:
            _fill_cache(cache, detector, x, active, ref_start, config)
            pvals = np.concatenate(
                [cache[(k, ref_start[k])].pvalue(s) for k, s in zip(active, stats)]
            )
            rejected = procedure.decide(pvals, rng)
        chunks.append(
            pd.DataFrame(
                {
                    "window": j,
                    "t_end": t1,
                    "stream": active,
                    "ref_start": ref_start[active],
                    "statistic": stats,
                    "pvalue": pvals,
                    "rejected": rejected,
                    "is_null": scenario.is_null(active, ref_start[active], t1),
                }
            )
        )
        ref_start[active[rejected]] = t1
    return MonitorResult(pd.concat(chunks, ignore_index=True), len(window_starts), scenario, config)


def _fill_cache(cache, detector, x, active, ref_start, config):
    missing = [k for k in active if (k, ref_start[k]) not in cache]
    if not missing:
        return
    n_ref = config.n_ref
    refs = np.stack([x[k, ref_start[k] : ref_start[k] + n_ref] for k in missing])
    seeds = [[config.calibration.seed, int(k), int(ref_start[k])] for k in missing]
    nulls = calibrate_many(detector, refs, config.window, config.calibration, seeds)
    for k, null in zip(missing, nulls):
        cache[(k, ref_start[k])] = null


def null_pvalues(
    scenario: Scenario,
    detector: Detector,
    config: MonitorConfig = MonitorConfig(),
) -> pd.DataFrame:
    """Statistics and p-values for every window against the initial reference, without alarms."""
    x = scenario.signal(detector.input_kind).astype(float)
    n_streams, n_steps = x.shape
    n_ref, W = config.n_ref, config.window
    refs = x[:, :n_ref]
    seeds = [[config.calibration.seed, k, 0] for k in range(n_streams)]
    nulls = calibrate_many(detector, refs, W, config.calibration, seeds)
    rows = []
    for j, t0 in enumerate(range(n_ref, n_steps - W + 1, W)):
        t1 = t0 + W
        stats = detector.window_statistic(np.concatenate([refs, x[:, t0:t1]], axis=1), n_ref)
        pvals = np.concatenate([null.pvalue(s) for null, s in zip(nulls, stats)])
        rows.append(
            pd.DataFrame(
                {
                    "window": j,
                    "t_start": t0,
                    "stream": np.arange(n_streams),
                    "statistic": stats,
                    "pvalue": pvals,
                    "is_null": scenario.is_null(np.arange(n_streams), 0, t1),
                    "block_length": [null.block_length for null in nulls],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)
