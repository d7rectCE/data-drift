"""Experiment 15: unified-FAR benchmark of every detector in the package.

As part B of experiment 1, with the stage-2 additions (sliding KS, persistent
MeanShift) and the stage-2 calibration (AR-sieve with parameter uncertainty,
moving blocks for 0/1 errors). Half of 1000 AR(1) streams (phi = 0.5) shift at
a known step by 0.5 or 1 sd; every detector is calibrated to a nominal FAR of
0.05 per window and compared by its actual FAR and its detection rate 1 and 3
windows after the shift.

Usage: python experiments/exp15_benchmark.py [--quick]
"""

from __future__ import annotations

import argparse

import pandas as pd

from common import N_REF, RESULTS, WINDOW, markdown_table, run_parallel
from driftfdr import (
    ADWIN,
    DDM,
    CalibrationConfig,
    KSSliding,
    KSWindow,
    MeanShift,
    MonitorConfig,
    PageHinkley,
    ScenarioConfig,
    make_scenario,
    run_monitor,
)

DETECTORS = {
    "PH": PageHinkley,
    "DDM": DDM,
    "ADWIN": ADWIN,
    "KS": KSWindow,
    "KS-sliding": KSSliding,
    "MeanShift(1)": lambda: MeanShift(1),
    "MeanShift(3)": lambda: MeanShift(3),
}


def task(args):
    name, magnitude, n = args
    onset = N_REF + 10 * WINDOW
    sc = make_scenario(ScenarioConfig(n_streams=n, n_steps=onset + 5 * WINDOW, phi=0.5, rho=0.0, drift_fraction=0.5,
                                      magnitude=magnitude, fixed_onset=onset), seed=1500)
    det = DETECTORS[name]()
    method = "moving" if det.input_kind == "errors" else "sieve_pu"
    t = run_monitor(sc, det, None, MonitorConfig(calibration=CalibrationConfig(method=method))).tests
    cal = t.pvalue <= 0.05
    row = {"detector": name, "magnitude": magnitude, "far": float(cal[t.is_null].mean())}
    for lag in (1, 3):
        post = (~t.is_null) & (t.t_end == onset + lag * WINDOW)
        row[f"power_w{lag}"] = float(cal[post].mean())
    return [row]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n = 100 if args.quick else 1000
    raw = run_parallel(task, [(d, m, n) for d in DETECTORS for m in (0.5, 1.0)])
    agg = raw.set_index(["detector", "magnitude"]).loc[[(d, m) for m in (0.5, 1.0) for d in DETECTORS]].reset_index()
    agg.to_csv(RESULTS / "exp15_benchmark.csv", index=False)
    with open(RESULTS / "exp15_tables.md", "w") as f:
        f.write("## Бенчмарк при едином номинальном FAR = 0.05 (φ = 0.5)\n\n" + markdown_table(agg) + "\n")
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
