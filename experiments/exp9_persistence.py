"""Experiment 9 (stage 2): requiring the degradation to persist.

After experiment 8 the remaining false retrains on Electricity were error
spikes that pass by themselves. ``MeanShift(persistence=m)`` alarms only if
the error is elevated in each of the last m windows. It is compared with
Page-Hinkley under the material-degradation null (delta = 5 points of error),
50 models, 3 fleets per data set.

Usage: python experiments/exp9_persistence.py [--quick]
"""

from __future__ import annotations

import argparse

import pandas as pd

from common import RESULTS, markdown_table, run_parallel
from driftfdr import CalibrationConfig, MeanShift, MonitorConfig, PageHinkley, make_procedure, run_monitor, summarize
from driftfdr.datasets import elec2_scenario, error_rate_view, insects_scenario

DETECTORS = {"PH": PageHinkley, "MeanShift(1)": lambda: MeanShift(1), "MeanShift(2)": lambda: MeanShift(2),
             "MeanShift(3)": lambda: MeanShift(3), "MeanShift(5)": lambda: MeanShift(5)}
PROCEDURES = ["uncorrected", "bonferroni", "bh_window"]
DELTA = 0.05


def task(args):
    data, seed, n_models = args
    base = insects_scenario(n_models=n_models, seed=seed) if data == "insects" else elec2_scenario(
        n_models=n_models, drift_fraction=0.2, flip=0.5, seed=seed)
    sc = error_rate_view(base, DELTA)
    cfg = MonitorConfig(calibration=CalibrationConfig(n_boot=500, tail="exponential", method="moving", tolerance=DELTA))
    rows = []
    for det_name, factory in DETECTORS.items():
        cache = {}
        for proc in PROCEDURES:
            s = summarize(run_monitor(sc, factory(), make_procedure(proc, 0.05), cfg, seed=seed, cache=cache))
            rows.append({"data": data, "detector": det_name, "procedure": proc, "seed": seed,
                         **{k: s[k] for k in ("alarms", "false_alarms", "true_alarms", "fdp", "power_per_test")}})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n_models, seeds = (12, 1) if args.quick else (50, 3)
    tasks = [(d, s, n_models) for d in ("elec2", "insects") for s in range(seeds)]
    raw = run_parallel(task, tasks)
    raw.to_csv(RESULTS / "exp9_runs.csv", index=False)
    agg = raw.groupby(["data", "detector", "procedure"])[["alarms", "false_alarms", "true_alarms", "fdp", "power_per_test"]].mean().reset_index()
    agg.to_csv(RESULTS / "exp9_persistence.csv", index=False)
    with open(RESULTS / "exp9_tables.md", "w") as f:
        f.write(f"## Устойчивость ухудшения, δ = {DELTA:g}, 50 моделей, 3 парка\n\n" + markdown_table(agg) + "\n")
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
