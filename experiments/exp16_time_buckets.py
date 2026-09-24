"""Experiment 16: measuring time in hours instead of rows (Airlines).

In experiment 14 the method did not help on Airlines: 1000 rows are about an
hour and a half of flights, and the delay rate follows the time of day (0.26 in
the morning, 0.53 in the evening). Here the 0/1 errors of each model are
averaged per hour (``bucket_means``) over all 31 days; the reference is one
week (168 hours), a window is one day (24 hours), and the truth is the error
over the next week. The same rules are compared with the row-level setting.

Usage: python experiments/exp16_time_buckets.py [--quick]
"""

from __future__ import annotations

import argparse

from common import RESULTS, markdown_table, run_parallel
from driftfdr import CalibrationConfig, MeanShift, MonitorConfig, PageHinkley, make_procedure, run_monitor, summarize
from driftfdr.datasets import airlines_hourly_scenario, error_rate_view

DELTA = 0.05
DETECTORS = {"PH": PageHinkley, "MeanShift(1)": lambda: MeanShift(1), "MeanShift(2)": lambda: MeanShift(2)}
PROCEDURES = ["uncorrected", "bonferroni", "bh_window"]


def task(args):
    seed, n = args
    sc = error_rate_view(airlines_hourly_scenario(n_models=n, seed=seed), DELTA, span=168)
    cfg = MonitorConfig(n_ref=168, window=24, horizon=3, calibration=CalibrationConfig(tolerance=DELTA))
    rows = []
    for name, factory in DETECTORS.items():
        cache = {}
        for proc in PROCEDURES:
            s = summarize(run_monitor(sc, factory(), make_procedure(proc, 0.05), cfg, seed=seed, cache=cache))
            rows.append({"detector": name, "procedure": proc, "seed": seed,
                         **{k: s[k] for k in ("alarms", "false_alarms", "fdp", "power_per_test")}})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n, seeds = (10, 1) if args.quick else (50, 3)
    raw = run_parallel(task, [(s, n) for s in range(seeds)])
    raw.to_csv(RESULTS / "exp16_runs.csv", index=False)
    agg = raw.groupby(["detector", "procedure"], sort=False)[["alarms", "false_alarms", "fdp", "power_per_test"]].mean().reset_index()
    agg.to_csv(RESULTS / "exp16_time_buckets.csv", index=False)
    with open(RESULTS / "exp16_tables.md", "w") as f:
        f.write("## Airlines, почасовые ошибки, опорный отрезок неделя, окно сутки, δ = 0.05, 50 моделей\n\n")
        f.write(markdown_table(agg) + "\n")
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
