"""Experiment 13: e-BH, FDR control that survives dependence between models.

Experiments 2 and 5 showed BH within a window losing FDR control when models
are correlated (false alarms come in bursts). e-BH controls FDR under any
dependence. It is compared with BH, Bonferroni and the uncorrected rule on
K = 200 models, 10% drifting, rho in {0, 0.3, 0.6}, Page-Hinkley, alpha = 0.05.

Usage: python experiments/exp13_ebh.py [--quick]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import RESULTS, markdown_table, run_parallel, run_procedures
from driftfdr import ScenarioConfig

PROCEDURES = ["uncorrected", "bonferroni", "bh_window", "e_bh"]
RHOS = [0.0, 0.3, 0.6]


def task(args):
    rho, seed, n = args
    cfg = ScenarioConfig(n_streams=n, n_steps=5000, phi=0.5, rho=rho, drift_fraction=0.1, magnitude=1.0)
    rows = run_procedures(cfg, seed, "PH", [(p, 0.05) for p in PROCEDURES])
    return [{"rho": rho, **r} for r in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n, seeds = (40, 2) if args.quick else (200, 10)
    raw = run_parallel(task, [(r, s, n) for r in RHOS for s in range(seeds)])
    raw.to_csv(RESULTS / "exp13_runs.csv", index=False)
    g = raw.groupby(["rho", "procedure"])
    agg = g[["fdp", "window_fdp", "false_alarms_per_window", "mdr", "degraded_per_drift"]].mean()
    agg["fdr_se"] = g.fdp.std() / np.sqrt(g.size())
    agg = agg.reset_index().rename(columns={"fdp": "fdr"})
    order = {p: i for i, p in enumerate(PROCEDURES)}
    agg = agg.sort_values(["rho", "procedure"], key=lambda s: s.map(order) if s.name == "procedure" else s)
    agg.to_csv(RESULTS / "exp13_ebh.csv", index=False)
    with open(RESULTS / "exp13_tables.md", "w") as f:
        f.write(f"## e-BH против BH и Бонферрони, K = {n}, α = 0.05\n\n" + markdown_table(agg) + "\n")
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
