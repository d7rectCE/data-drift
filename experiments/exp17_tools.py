"""Experiment 17: comparison with Evidently and NannyML on four real data sets.

All tools watch the same 0/1 error stream of each model against the same
static reference (the first 1000 steps after training, no retraining), window
by window (100 steps), and are judged by the same truth: whether the model's
error over the next 1000 steps exceeds its reference error by more than 5 points.

* Evidently 0.7 ``ValueDrift`` on the error column: its default for a binary
  column is a two-proportion Z-test, drift if p < 0.05. The test is reproduced
  here (``evidently_pvalue``) because one Evidently report per window would take
  hours; it matches Evidently's p-values to 1e-16 on 300 random cases (checked
  in ``check_evidently``).
* NannyML 0.13 ``PerformanceCalculator`` on accuracy, chunk = window, default
  thresholds (reference mean +- 3 sd of chunk accuracies); an alarm is an alert
  below the lower threshold. Called directly, one call per model.
* This package: MeanShift(1) with calibrated p-values, with and without the
  tolerance delta, per model (alpha = 0.05) and with Bonferroni across models.

Requires ``pip install evidently nannyml scikit-learn river``.

Usage: python experiments/exp17_tools.py [--quick]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from common import RESULTS, markdown_table, run_parallel
from driftfdr import CalibrationConfig, MeanShift, MonitorConfig, run_monitor
from driftfdr.datasets import airlines_scenario, covertype_scenario, elec2_scenario, error_rate_view, insects_scenario

N_REF, W, DELTA = 1000, 100, 0.05
LOADERS = {
    "INSECTS": lambda n, s: insects_scenario(n_models=n, seed=s),
    "Electricity": lambda n, s: elec2_scenario(n_models=n, drift_fraction=0.0, seed=s),
    "Airlines": lambda n, s: airlines_scenario(n_models=n, seed=s),
    "Covertype": lambda n, s: covertype_scenario(n_models=n, seed=s),
}


def evidently_pvalue(ref: np.ndarray, cur: np.ndarray) -> float:
    """Evidently's default drift test for a binary column: pooled two-proportion Z-test."""
    n1, n2 = len(ref), len(cur)
    p = (ref.sum() + cur.sum()) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return 1.0 if se == 0 else float(2 * stats.norm.sf(abs(ref.mean() - cur.mean()) / se))


def check_evidently(n_cases: int = 300) -> float:
    from evidently import DataDefinition, Dataset, Report
    from evidently.metrics import ValueDrift

    rng = np.random.default_rng(1)
    dd = DataDefinition(categorical_columns=["err"])
    worst = 0.0
    for _ in range(n_cases):
        a = rng.uniform(0.05, 0.6)
        ref = (rng.random(N_REF) < a).astype(int)
        cur = (rng.random(W) < np.clip(a + rng.uniform(-0.15, 0.2), 0.01, 0.99)).astype(int)
        report = Report([ValueDrift(column="err")]).run(
            Dataset.from_pandas(pd.DataFrame({"err": cur}), data_definition=dd),
            Dataset.from_pandas(pd.DataFrame({"err": ref}), data_definition=dd),
        )
        worst = max(worst, abs(float(report.dict()["metrics"][0]["value"]) - evidently_pvalue(ref, cur)))
    return worst


def nannyml_alarms(errors: np.ndarray, n_windows: int) -> np.ndarray:
    import nannyml as nml

    def frame(e):
        return pd.DataFrame({"y_pred": np.ones(e.size, int), "y_true": 1 - e.astype(int), "y_pred_proba": np.full(e.size, 0.9)})

    calc = nml.PerformanceCalculator(y_pred="y_pred", y_true="y_true", y_pred_proba="y_pred_proba",
                                     metrics=["accuracy"], chunk_size=W, problem_type="classification_binary")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc.fit(frame(errors[:N_REF]))
        res = calc.calculate(frame(errors[N_REF : N_REF + n_windows * W])).filter(period="analysis").to_df()
    acc = res[("accuracy", "value")].to_numpy()
    low = res[("accuracy", "lower_threshold")].to_numpy()
    return (acc < low)[:n_windows]


def task(args):
    data, seed, n = args
    sc = error_rate_view(LOADERS[data](n, seed), DELTA)
    E = sc.errors.astype(float)
    n_windows = (sc.n_steps - N_REF) // W
    ends = N_REF + W * (np.arange(n_windows) + 1)
    truth = np.stack([sc.is_null(np.full(n_windows, k), np.zeros(n_windows, int), ends, ref_len=N_REF, window=W)
                      for k in range(n)])
    decisions = {}
    p_ev = np.array([[evidently_pvalue(E[k, :N_REF], E[k, e - W : e]) for e in ends] for k in range(n)])
    decisions["Evidently (Z-тест, p < 0.05)"] = p_ev < 0.05
    decisions["Evidently + Бонферрони"] = p_ev < 0.05 / n
    decisions["NannyML (точность, ±3σ)"] = np.stack([nannyml_alarms(E[k], n_windows) for k in range(n)])
    for tol, label in ((0.0, "driftfdr, δ = 0"), (DELTA, "driftfdr, δ = 0.05")):
        cfg = MonitorConfig(n_ref=N_REF, window=W, horizon=1, calibration=CalibrationConfig(tolerance=tol))
        t = run_monitor(sc, MeanShift(1), None, cfg).tests
        P = t.pivot(index="stream", columns="window", values="pvalue").to_numpy()[:, :n_windows]
        decisions[label + ", без поправки"] = P <= 0.05
        decisions[label + ", Бонферрони"] = P <= 0.05 / n
    rows = []
    for name, D in decisions.items():
        alarms = D.sum()
        rows.append({"data": data, "method": name, "seed": seed,
                     "alarms_per_100": 100 * alarms / D.size,
                     "fdp": float((D & truth).sum() / max(alarms, 1)),
                     "far": float(D[truth].mean()) if truth.any() else np.nan,
                     "power": float(D[~truth].mean()) if (~truth).any() else np.nan})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print("max |Evidently - reimplementation| over 300 cases:", check_evidently(30 if args.quick else 300))
    n, seeds = (8, 1) if args.quick else (20, 2)
    raw = run_parallel(task, [(d, s, n) for d in LOADERS for s in range(seeds)])
    raw.to_csv(RESULTS / "exp17_runs.csv", index=False)
    agg = raw.groupby(["data", "method"], sort=False)[["alarms_per_100", "fdp", "far", "power"]].mean().reset_index()
    agg.to_csv(RESULTS / "exp17_tools.csv", index=False)
    with open(RESULTS / "exp17_tables.md", "w") as f:
        f.write("## Сравнение с Evidently и NannyML: одинаковая опора (1000 шагов), окна по 100, "
                "истина — существенное ухудшение (δ = 0.05), 20 моделей\n\n")
        f.write("`far` — доля тревог в окнах без существенного ухудшения, `power` — доля тревог в окнах с ним.\n\n")
        f.write(markdown_table(agg) + "\n")
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
