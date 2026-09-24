"""Experiment 10: more replicates or a heavier tail model for the deep tail.

Experiment 6 left the per-window FWER of Bonferroni at 0.077 (PH) and 0.18
(ADWIN) against 0.05. Two further levers are compared on 100 independent
AR(1) streams (phi = 0.5), AR-sieve with parameter uncertainty throughout:

* B = 2000 bootstrap replicates instead of 500 (less extrapolation);
* a generalised Pareto tail with estimated shape instead of an exponential one.

Usage: python experiments/exp10_tail_shape.py [--quick]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import RESULTS, markdown_table, make_detector, run_parallel
from driftfdr import CalibrationConfig, MonitorConfig, ScenarioConfig, make_scenario, run_monitor

CONFIGS = {"B=500, exp": (500, "exponential"), "B=500, GPD": (500, "gpd"),
           "B=2000, exp": (2000, "exponential"), "B=2000, GPD": (2000, "gpd")}


def task(args):
    det, label, n_streams, seed = args
    n_boot, tail = CONFIGS[label]
    sc = make_scenario(ScenarioConfig(n_streams=n_streams, n_steps=5000, phi=0.5, rho=0.0, drift_fraction=0.0), seed=700 + seed)
    cfg = MonitorConfig(calibration=CalibrationConfig(n_boot=n_boot, method="sieve_pu", tail=tail))
    t = run_monitor(sc, make_detector(det), None, cfg).tests
    min_p = t.groupby("window").pvalue.min().to_numpy()
    p = t.pvalue.to_numpy()
    return [{"detector": det, "config": label, "seed": seed, "fwer": float(np.mean(min_p <= 0.05 / n_streams)),
             "far@1e-3": float(np.mean(p <= 1e-3)), "far@1e-4": float(np.mean(p <= 1e-4))}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n_streams, seeds = (50, 2) if args.quick else (100, 30)
    tasks = [(d, c, n_streams, s) for d in ("ADWIN", "PH", "KS") for c in CONFIGS for s in range(seeds)]
    raw = run_parallel(task, tasks)
    raw.to_csv(RESULTS / "exp10_runs.csv", index=False)
    g = raw.groupby(["detector", "config"])
    agg = pd.DataFrame({"fwer": g.fwer.mean(), "se": g.fwer.std() / np.sqrt(g.size()),
                        "far@1e-3": g["far@1e-3"].mean(), "far@1e-4": g["far@1e-4"].mean()}).reset_index()
    agg.to_csv(RESULTS / "exp10_tail_shape.csv", index=False)
    with open(RESULTS / "exp10_tables.md", "w") as f:
        f.write(f"## Бонферрони, P(ложная тревога в окне), K = {n_streams}, цель 0.05\n\n" + markdown_table(agg) + "\n")
    print(agg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
