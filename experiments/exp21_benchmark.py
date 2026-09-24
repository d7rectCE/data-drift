"""Experiment 21: the fixed benchmark of 100 synthetic scenarios.

``benchmark_suite()`` gives 25 cases x 4 seeds: abrupt or gradual drift of 0.5
or 1 standard deviation, correlation between models rho = 0, 0.3, 0.6,
scattered drifts (10% of 100 models) or two drift events (5% each), plus a case
without drift. Every calibrated detector runs with every main rule at
alpha = 0.05, on the raw streams and on the residuals of ``split_common``
(experiment 18); river's Page-Hinkley at its default threshold is the baseline.

Metrics are those of ``summarize``: false alarms per 1000 model-steps, the
probability of a false alarm in a window, FDP, event-level precision, recall
and F1 (a detection counts within 1000 steps of the change), delay and steps
spent on a degraded model per drift.

Usage: python experiments/exp21_benchmark.py [--quick]
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from common import RESULTS, default_resampling, make_detector, markdown_table, monitor_config, run_parallel
from driftfdr import MeanShift, RawThreshold, benchmark_suite, make_procedure, make_scenario, run_monitor, split_common, summarize
from driftfdr.streams import NO_CHANGE

ALPHA, MAX_DELAY, N_REF = 0.05, 1000, 300
DETECTORS = {
    "PH": lambda: make_detector("PH"),
    "KS": lambda: make_detector("KS"),
    "ADWIN": lambda: make_detector("ADWIN"),
    "MeanShift(1)": lambda: MeanShift(1),
    "MeanShift(3)": lambda: MeanShift(3),
}
RULES = ["uncorrected", "bonferroni", "bh_window", "LORD++"]
METRICS = ["false_alarms_per_1k_stream_steps", "p_any_false_alarm_per_window", "fdp", "precision", "recall", "f1",
           "mean_delay", "degraded_per_drift"]


def task(args):
    case, cfg, seed = args
    sc = make_scenario(cfg, seed)
    views = {"сырые": sc, "остатки": replace(sc, values=split_common(sc.values, N_REF)[0])}
    rows = []
    for det_name, factory in DETECTORS.items():
        det = factory()
        mc = monitor_config(default_resampling(det))
        for view, data in views.items():
            cache = {}
            for rule in RULES:
                if view == "остатки" and rule in ("uncorrected", "LORD++"):
                    continue
                res = run_monitor(data, det, make_procedure(rule, ALPHA), mc, seed=seed, cache=cache)
                rows.append({"detector": det_name, "view": view, "rule": rule, **summarize(res, max_delay=MAX_DELAY)})
    ph = make_detector("PH")
    res = run_monitor(sc, ph, RawThreshold(ph.default_threshold), monitor_config("sieve"), seed=seed)
    rows.append({"detector": "PH river по умолчанию", "view": "сырые", "rule": "порог", **summarize(res, max_delay=MAX_DELAY)})
    kind, pattern = cfg.drift_type, "clustered" if cfg.drift_events else "scattered"
    if cfg.drift_fraction == 0 and not cfg.drift_events:
        kind, pattern = "none", "none"
    return [{"case": case, "seed": seed, "drift_type": kind, "magnitude": cfg.magnitude, "rho": cfg.rho,
             "pattern": pattern, **{k: r[k] for k in ("detector", "view", "rule")}, **{m: r[m] for m in METRICS}}
            for r in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    suite = benchmark_suite(n_streams=30, seeds_per_case=1)[::6] if args.quick else benchmark_suite()
    raw = run_parallel(task, suite)
    raw.to_csv(RESULTS / "exp21_runs.csv", index=False)

    keys = ["detector", "view", "rule"]
    overall = raw.groupby(keys, sort=False)[METRICS].mean().reset_index().sort_values("f1", ascending=False)
    overall.to_csv(RESULTS / "exp21_overall.csv", index=False)
    drift = raw[raw.drift_type != "none"]
    by = {}
    for factor in ("drift_type", "magnitude", "rho", "pattern"):
        t = drift.groupby(keys + [factor])["f1"].mean().unstack(factor)
        t.columns = [f"{factor} = {c}" for c in t.columns]
        by[factor] = t.loc[overall.set_index(keys).index.intersection(t.index)].reset_index()
    null_case = raw[raw.drift_type == "none"].groupby(keys, sort=False)[
        ["false_alarms_per_1k_stream_steps", "p_any_false_alarm_per_window"]].mean().reset_index()

    with open(RESULTS / "exp21_tables.md", "w") as f:
        f.write(f"## Бенчмарк: {raw.case.nunique()} случаев × {raw.seed.nunique() // raw.case.nunique()} сидов, "
                f"α = {ALPHA}, F1 при задержке ≤ {MAX_DELAY} шагов\n\n### Все сценарии, по убыванию F1\n\n"
                + markdown_table(overall) + "\n\n### Сценарий без дрейфа: ложные тревоги\n\n" + markdown_table(null_case))
        for factor, t in by.items():
            f.write(f"\n\n### F1 по фактору `{factor}`\n\n" + markdown_table(t))
        f.write("\n")
    print(overall.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
