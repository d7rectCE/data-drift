"""Experiment 14: four real data sets under the material-degradation null.

INSECTS, Electricity (natural data only, no injected drift), Airlines and
Covertype, each monitored by a fleet of 50 models in original row order.
Alarms are judged by whether the model's error over the next 1000 steps
exceeds its reference error by more than delta = 5 points, which needs no
drift labels. River's Page-Hinkley at its default threshold on the 0/1 error
stream is the "current practice" baseline.

Usage: python experiments/exp14_real_all.py [--quick]
"""

from __future__ import annotations

import argparse

import numpy as np

from common import RESULTS, markdown_table, run_parallel
from driftfdr import CalibrationConfig, MeanShift, MonitorConfig, PageHinkley, RawThreshold, make_procedure, run_monitor, summarize
from driftfdr.datasets import airlines_scenario, covertype_scenario, elec2_scenario, error_rate_view, insects_scenario

DELTA = 0.05
LOADERS = {
    "INSECTS": lambda n, s: insects_scenario(n_models=n, seed=s),
    "Electricity": lambda n, s: elec2_scenario(n_models=n, drift_fraction=0.0, seed=s),
    "Airlines": lambda n, s: airlines_scenario(n_models=n, seed=s),
    "Covertype": lambda n, s: covertype_scenario(n_models=n, seed=s),
}
RULES = [("PH", "raw"), ("PH", "uncorrected"), ("PH", "bonferroni"), ("PH", "bh_window"),
         ("MeanShift(3)", "uncorrected"), ("MeanShift(3)", "bonferroni"), ("MeanShift(3)", "bh_window")]


def task(args):
    data, seed, n = args
    sc = error_rate_view(LOADERS[data](n, seed), DELTA)
    cfg = MonitorConfig(calibration=CalibrationConfig(n_boot=500, tail="exponential", method="moving", tolerance=DELTA))
    caches = {"PH": {}, "MeanShift(3)": {}}
    rows = []
    for det_name, rule in RULES:
        det = PageHinkley() if det_name == "PH" else MeanShift(3)
        proc = RawThreshold(det.default_threshold) if rule == "raw" else make_procedure(rule, 0.05)
        s = summarize(run_monitor(sc, det, proc, cfg, seed=seed, cache=caches[det_name]))
        rows.append({"data": data, "detector": det_name, "rule": rule, "seed": seed,
                     "alarms_per_model_10k": 1e4 * s["alarms"] / (n * sc.n_steps),
                     **{k: s[k] for k in ("alarms", "false_alarms", "fdp", "power_per_test")}})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n, seeds = (10, 1) if args.quick else (50, 2)
    raw = run_parallel(task, [(d, s, n) for d in LOADERS for s in range(seeds)])
    raw.to_csv(RESULTS / "exp14_runs.csv", index=False)
    agg = raw.groupby(["data", "detector", "rule"], sort=False)[["alarms_per_model_10k", "fdp", "power_per_test"]].mean().reset_index()
    agg.to_csv(RESULTS / "exp14_real_all.csv", index=False)
    with open(RESULTS / "exp14_tables.md", "w") as f:
        f.write(f"## Четыре реальных набора, истина — существенное ухудшение (δ = {DELTA:g}), 50 моделей\n\n")
        f.write("`raw` — Page-Hinkley river с порогом по умолчанию на ряде ошибок 0/1, без калибровки.\n\n")
        f.write(markdown_table(agg) + "\n")
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
