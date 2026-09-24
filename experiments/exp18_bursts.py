"""Experiment 18: telling bursts of false alarms from real clustered drift.

When models share a common factor, many of them look worse at once even
without any drift (experiment 5), just as they do after a real drift event
(experiment 4). ``split_common`` removes the cross-sectional median from the
standardised streams: model-specific residuals are tested per model, and the
common component is tested as one extra stream; if it alarms, the whole fleet
is flagged. Compared with testing the raw streams directly, on K = 200 AR(1)
streams (phi = 0.5, rho in {0.3, 0.6}) with no drift, a drift event hitting
20% of the models, or a drift hitting all of them. Page-Hinkley, Bonferroni
or BH within each window, alpha = 0.05.

Usage: python experiments/exp18_bursts.py [--quick]
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from common import RESULTS, markdown_table, run_parallel
from driftfdr import (
    CalibrationConfig,
    MonitorConfig,
    PageHinkley,
    ScenarioConfig,
    make_procedure,
    make_scenario,
    run_monitor,
    split_common,
    summarize,
)
from driftfdr.streams import NO_CHANGE, Scenario

N_REF = 300
SCENARIOS = {
    "без дрейфа": dict(drift_fraction=0.0),
    "событие, 20% моделей": dict(drift_fraction=0.0, drift_events=1, event_fraction=0.2),
    "событие, все модели": dict(drift_fraction=0.0, drift_events=1, event_fraction=1.0),
}
CFG = MonitorConfig(n_ref=N_REF, calibration=CalibrationConfig(n_boot=500, tail="exponential"))


def augmented(sc: Scenario, resid: np.ndarray, common: np.ndarray) -> Scenario:
    """Residual streams plus the common component as stream K; the latter changes iff most streams do."""
    n = sc.n_streams
    onset = int(np.min(sc.change_start)) if sc.drifting.size > n / 2 else NO_CHANGE
    return replace(
        sc,
        values=np.vstack([resid, common[None]]),
        errors=np.zeros((n + 1, sc.n_steps), np.int8),
        change_start=np.append(sc.change_start, onset),
        change_end=np.append(sc.change_end, onset),
        drift_kind=np.append(sc.drift_kind, "fleet"),
        event=np.append(sc.event, -1),
        later_changes=None,
        mean_shift=None,
    )


def model_metrics(t: pd.DataFrame, sc: Scenario, fleet_times: np.ndarray, n_windows: int) -> dict:
    """Per-model alarms; a true fleet alarm also counts as catching every drift it covers."""
    rej, null = t.rejected.to_numpy(), t.is_null.to_numpy()
    per_window_fa = t[rej & null].groupby("window").size()
    delays = []
    for k in sc.drifting:
        onset = sc.change_start[k]
        hits = list(t.t_end[(t.stream == k) & rej & (t.t_end > onset)]) + [f for f in fleet_times if f > onset]
        delays.append((min(hits) if hits else sc.n_steps) - onset)
    return {"false_alarms": int((rej & null).sum()), "fdp": float((rej & null).sum() / max(rej.sum(), 1)),
            "p_any_false_alarm_per_window": per_window_fa.size / n_windows,
            "max_burst": int(per_window_fa.max()) if not per_window_fa.empty else 0,
            "degraded_per_drift": float(np.mean(delays)) if delays else np.nan}


def task(args):
    scen, rho, seed, n, rule = args
    sc = make_scenario(ScenarioConfig(n_streams=n, n_steps=5000, phi=0.5, rho=rho, magnitude=1.0, **SCENARIOS[scen]),
                       seed=seed)
    det = PageHinkley()
    rows = []
    res = run_monitor(sc, det, make_procedure(rule, 0.05), CFG, seed=seed)
    rows.append({"method": "сырые ряды", **model_metrics(res.tests, sc, np.array([]), res.n_windows),
                 "fleet_true": 0, "fleet_false": 0})
    resid, common = split_common(sc.values, N_REF)
    aug = augmented(sc, resid, common)
    for label, with_fleet in (("остатки", False), ("остатки + тест парка", True)):
        view = aug if with_fleet else replace(sc, values=resid)
        r = run_monitor(view, det, make_procedure(rule, 0.05), CFG, seed=seed)
        t = r.tests
        fleet = t[(t.stream == n) & t.rejected] if with_fleet else t.iloc[:0]
        models = t[t.stream < n]
        rows.append({"method": label, **model_metrics(models, sc, fleet.t_end.to_numpy(), r.n_windows),
                     "fleet_true": int((~fleet.is_null).sum()), "fleet_false": int(fleet.is_null.sum())})
    return [{"rule": rule, "scenario": scen, "rho": rho, "seed": seed, **row} for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n, seeds = (60, 2) if args.quick else (200, 10)
    tasks = [(sc, rho, s, n, rule) for rule in ("bonferroni", "bh_window") for sc in SCENARIOS
             for rho in (0.3, 0.6) for s in range(seeds)]
    raw = run_parallel(task, tasks)
    raw.to_csv(RESULTS / "exp18_runs.csv", index=False)
    cols = ["false_alarms", "fdp", "p_any_false_alarm_per_window", "max_burst", "degraded_per_drift", "fleet_true", "fleet_false"]
    agg = raw.groupby(["rule", "scenario", "rho", "method"], sort=False)[cols].mean().reset_index()
    agg.to_csv(RESULTS / "exp18_bursts.csv", index=False)
    with open(RESULTS / "exp18_tables.md", "w") as f:
        f.write(f"## Пачки ложных тревог против массового дрейфа, K = {n}, Page-Hinkley, Бонферрони и BH в окне\n\n")
        f.write("`max_burst` — наибольшее число ложных тревог моделей в одном окне; `fleet_true` / `fleet_false` — "
                "верные и ложные тревоги теста общего компонента (событие по всему парку).\n\n" + markdown_table(agg) + "\n")
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
