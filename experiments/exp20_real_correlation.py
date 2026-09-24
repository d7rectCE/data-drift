"""Experiment 20: how correlated are the models' p-values on real data?

Experiment 5 found that a dependence-aware correction (joint resampling of the
streams) can beat Bonferroni only when the models' tests are strongly
correlated. This measures that correlation on the four real data sets and
places them on the synthetic rho scale of experiments 5, 13 and 18.

For every model, MeanShift(1) turns each window of 100 steps of the 0/1 error
stream into a p-value against a static reference (the first 1000 steps after
training, as in experiment 17; delta = 0, no retraining). Only windows without
material degradation (delta = 5 points, forward oracle) are used, so the
correlation is that of the null p-values, which is what a multiplicity
correction has to deal with. Two numbers per data set:

* the mean Spearman correlation of p-values between pairs of models, over the
  windows where both are null;
* the dispersion index (variance / mean over windows) of the number of models
  with p < 0.05 among the null ones: 1 without dependence, larger when false
  alarms come in bursts (experiment 5).

The same numbers after removing the common component with ``split_common``
(experiment 18) show how much of the dependence it takes away, and the same
numbers on synthetic AR(1) fleets (phi = 0.5) with a common factor of
share rho give the scale.

Usage: python experiments/exp20_real_correlation.py [--quick]
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import RESULTS, markdown_table, run_parallel
from driftfdr import CalibrationConfig, MeanShift, MonitorConfig, ScenarioConfig, make_scenario, run_monitor, split_common
from driftfdr.datasets import airlines_scenario, covertype_scenario, elec2_scenario, error_rate_view, insects_scenario

DELTA, N_REF, W = 0.05, 1000, 100
LOADERS = {
    "INSECTS": lambda n, s: insects_scenario(n_models=n, seed=s),
    "Electricity": lambda n, s: elec2_scenario(n_models=n, drift_fraction=0.0, seed=s),
    "Airlines": lambda n, s: airlines_scenario(n_models=n, seed=s),
    "Covertype": lambda n, s: covertype_scenario(n_models=n, seed=s),
}
RHOS = (0.0, 0.3, 0.6, 0.9)


def null_pvalues(sc, seed):
    """(models x windows) p-values and null mask, static reference, no retraining."""
    cfg = MonitorConfig(n_ref=N_REF, window=W, horizon=1, calibration=CalibrationConfig())
    t = run_monitor(sc, MeanShift(1), None, cfg, seed=seed).tests
    P = t.pivot(index="stream", columns="window", values="pvalue").to_numpy()
    N = t.pivot(index="stream", columns="window", values="is_null").to_numpy().astype(bool)
    return P, N


def dependence(P, N):
    n = P.shape[0]
    corrs = []
    for i in range(n):
        for j in range(i + 1, n):
            both = N[i] & N[j]
            if both.sum() >= 20 and np.ptp(P[i, both]) > 0 and np.ptp(P[j, both]) > 0:
                corrs.append(spearmanr(P[i, both], P[j, both])[0])
    counts = ((P < 0.05) & N).sum(axis=0)
    full = N.mean(axis=0) >= 0.8  # windows where most models are null
    c = counts[full]
    return {"spearman": float(np.nanmean(corrs)) if corrs else np.nan,
            "dispersion": float(c.var() / c.mean()) if c.size and c.mean() > 0 else np.nan,
            "null_share": float(N.mean()), "windows": int(full.sum())}


def task(args):
    kind, key, seed, n = args
    if kind in ("real", "residuals"):
        sc = error_rate_view(LOADERS[key](n, seed), DELTA)
        if kind == "residuals":  # the common component removed as in experiment 18
            sc = replace(sc, values=split_common(sc.values, N_REF)[0])
        P, N = null_pvalues(sc, seed)
    else:  # continuous AR(1) streams without drift: every window is null
        sc = make_scenario(ScenarioConfig(n_streams=n, n_steps=N_REF + 300 * W, phi=0.5, rho=key, drift_fraction=0.0),
                           seed=seed)
        cfg = MonitorConfig(n_ref=N_REF, window=W, horizon=1, calibration=CalibrationConfig())
        t = run_monitor(sc, MeanShift(1), None, cfg, seed=seed).tests
        P = t.pivot(index="stream", columns="window", values="pvalue").to_numpy()
        N = np.ones_like(P, dtype=bool)
    return [{"source": kind, "data": f"синтетика, ρ = {key}" if kind == "synthetic" else key, "seed": seed,
             **dependence(P, N)}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n, seeds = (10, 1) if args.quick else (50, 2)
    tasks = [("real", d, s, n) for d in LOADERS for s in range(seeds)]
    tasks += [("residuals", d, s, n) for d in LOADERS for s in range(seeds)]
    tasks += [("synthetic", rho, s, n) for rho in RHOS for s in range(seeds)]
    raw = run_parallel(task, tasks)
    raw.to_csv(RESULTS / "exp20_runs.csv", index=False)
    agg = raw.groupby(["source", "data"], sort=False)[["spearman", "dispersion", "null_share", "windows"]].mean().reset_index()
    order = {"real": 0, "residuals": 1, "synthetic": 2}
    agg = agg.sort_values(["source", "data"], key=lambda c: c.map(order) if c.name == "source" else c, kind="stable")
    agg.to_csv(RESULTS / "exp20_correlation.csv", index=False)
    with open(RESULTS / "exp20_tables.md", "w") as f:
        f.write(f"## Зависимость p-значений между моделями, {n} моделей, MeanShift(1), окна по {W}\n\n"
                "`spearman` — средняя корреляция p-значений пар моделей по нулевым окнам; `dispersion` — "
                "дисперсия / среднее числа моделей с p < 0.05 в окне (1 без зависимости); `null_share` — "
                "доля нулевых окон.\n\n" + markdown_table(agg) + "\n")
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
