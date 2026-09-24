"""Evaluation of a monitoring run against the known change points.

Error side: FDP (false alarms / alarms over the whole run), the per-window
FDP averaged over windows (what a per-window BH controls), false alarms per
window, probability of at least one false alarm in a window, and MTFA in
stream-steps.
Detection side: for every drifting stream the delay is the time from the onset
to the end of the first window whose test alarms; a drift that is never
caught counts as missed (MDR) and keeps degrading the model until the end of
the run. ``degraded_per_drift`` averages this censored delay over all drifts,
so it penalises both slow detection and misses.
"""

from __future__ import annotations

import numpy as np

from .monitor import MonitorResult
from .streams import NO_CHANGE


def summarize(result: MonitorResult) -> dict:
    tests = result.tests
    scenario = result.scenario
    n_streams, n_steps = scenario.n_streams, scenario.n_steps
    rejected = tests["rejected"].to_numpy()
    null = tests["is_null"].to_numpy()
    n_alarms = int(rejected.sum())
    n_false = int((rejected & null).sum())
    fa_windows = tests.loc[rejected & null, "window"].nunique()
    per_window = tests.assign(v=rejected & null).groupby("window")[["v", "rejected"]].sum()
    window_fdp = (per_window["v"] / per_window["rejected"].clip(lower=1)).sum() / result.n_windows
    monitored_steps = n_streams * (n_steps - result.config.n_ref)

    alarms = tests.loc[rejected, ["stream", "t_end"]]
    alarm_times = {k: np.sort(g.to_numpy()) for k, g in alarms.groupby("stream")["t_end"]}
    cs, _ = scenario.changes()
    delays, degraded = [], []
    for k in scenario.drifting:
        onsets = np.sort(cs[k][cs[k] < NO_CHANGE])
        times = alarm_times.get(k, np.array([], dtype=int))
        for i, onset in enumerate(onsets):
            # a change counts as caught by the first alarm after it and before the next change
            until = onsets[i + 1] if i + 1 < len(onsets) else n_steps
            hit = times[(times > onset) & (times <= until)]
            if hit.size:
                delays.append(hit[0] - onset)
                degraded.append(hit[0] - onset)
            else:
                degraded.append(until - onset)
    n_drifts = len(degraded)

    return {
        "n_tests": len(tests),
        "n_null_tests": int(null.sum()),
        "alarms": n_alarms,
        "false_alarms": n_false,
        "true_alarms": n_alarms - n_false,
        "fdp": n_false / max(n_alarms, 1),
        "window_fdp": float(window_fdp),
        "far_per_test": n_false / max(int(null.sum()), 1),
        "power_per_test": int((rejected & ~null).sum()) / max(int((~null).sum()), 1),
        "false_alarms_per_window": n_false / result.n_windows,
        "p_any_false_alarm_per_window": fa_windows / result.n_windows,
        "false_alarms_per_1k_stream_steps": 1000 * n_false / monitored_steps,
        "mtfa_stream_steps": monitored_steps / n_false if n_false else np.inf,
        "n_drifts": n_drifts,
        "detected": len(delays),
        "mdr": 1 - len(delays) / n_drifts if n_drifts else np.nan,
        "mean_delay": float(np.mean(delays)) if delays else np.nan,
        "median_delay": float(np.median(delays)) if delays else np.nan,
        "degraded_per_drift": float(np.mean(degraded)) if degraded else np.nan,
    }
