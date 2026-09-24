"""Experiment 12: trade-off by drift type and intensity.

Trade-off curves over alpha (as in experiment 3) for abrupt and gradual
(ramp of 500 steps) drifts of 0.5 and 1 standard deviation, K = 100 models,
10% drifting, Page-Hinkley. Rules are compared by the best delay reached
within a false-retrain budget, and by MTR at alpha = 0.05.

Usage: python experiments/exp12_drift_types.py [--quick]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import RESULTS, markdown_table, run_parallel, run_procedures
from driftfdr import ScenarioConfig

PROCEDURES = ["uncorrected", "bonferroni", "bh_window", "LORD++"]
ALPHAS = [1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1]
CASES = {f"{t}, Δ={m:g}": (t, m) for t in ("abrupt", "gradual") for m in (0.5, 1.0)}
BUDGETS = [0.001, 0.01]


def task(args):
    case, seed, n = args
    kind, mag = CASES[case]
    cfg = ScenarioConfig(n_streams=n, n_steps=5000, phi=0.5, rho=0.3, drift_fraction=0.1, drift_type=kind,
                         magnitude=mag, gradual_length=500)
    rows = run_procedures(cfg, seed, "PH", [(p, a) for p in PROCEDURES for a in ALPHAS])
    return [{"case": case, **r} for r in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n, seeds = (30, 1) if args.quick else (100, 6)
    raw = run_parallel(task, [(c, s, n) for c in CASES for s in range(seeds)])
    raw.to_csv(RESULTS / "exp12_runs.csv", index=False)
    cols = ["false_alarms_per_1k_stream_steps", "degraded_per_drift", "mdr", "mean_delay", "mtr", "fdp"]
    agg = raw.groupby(["case", "procedure", "alpha"])[cols].mean().reset_index()
    agg.to_csv(RESULTS / "exp12_tradeoff.csv", index=False)
    best = []
    for (case, proc), g in agg.groupby(["case", "procedure"]):
        for b in BUDGETS:
            ok = g[g.false_alarms_per_1k_stream_steps <= b]
            best.append({"case": case, "procedure": proc, "budget": b,
                         "degraded_per_drift": ok.degraded_per_drift.min() if not ok.empty else np.nan})
    best = pd.DataFrame(best)
    table = best.pivot_table(index=["case", "budget"], columns="procedure", values="degraded_per_drift").reset_index()
    at05 = agg[agg.alpha == 0.05].pivot(index="case", columns="procedure", values="mtr").reset_index()
    with open(RESULTS / "exp12_tables.md", "w") as f:
        f.write("## Лучшая задержка (шагов на старой модели) в пределах бюджета ложных переобучений\n\n")
        f.write(markdown_table(table) + "\n\n## MTR при α = 0.05 (больше — лучше)\n\n" + markdown_table(at05) + "\n")
    print(table.round(0).to_string(index=False))
    print(at05.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
