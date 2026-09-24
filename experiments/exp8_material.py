"""Experiment 8 (stage 2): a null of material degradation instead of "no change".

Experiment 7 failed on Electricity because the data change all the time, so
"no drift" never holds. For a retraining decision what matters is whether the
model got materially worse. Two changes follow:

* the test: H0 is "the monitored error rose by at most delta"; calibration
  shifts the bootstrap continuation up by delta (``CalibrationConfig.tolerance``);
* the evaluation: ground truth is what the model will cost if not retrained,
  known in hindsight: its error rate over the next 1000 steps from the tested
  window, compared with its error rate over the reference. A test is null iff
  the rise is <= delta, so transient spikes that pass by themselves are null.

Part A: operating characteristic on synthetic streams whose shifts are drawn
uniformly in [0, 1] sd, so part of the drifts are below the tolerance.
Part B: INSECTS and Electricity under the material null, delta = 5 points of
error rate, Page-Hinkley on the 0/1 error stream.

Usage: python experiments/exp8_material.py [--quick]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import (
    PALETTE,
    RESULTS,
    TEXT_2,
    line_kwargs,
    make_detector,
    markdown_table,
    monitor_config,
    run_parallel,
    savefig,
    setup_style,
)
from driftfdr import CalibrationConfig, MonitorConfig, ScenarioConfig, make_procedure, make_scenario, run_monitor, summarize
from driftfdr.datasets import elec2_scenario, error_rate_view, insects_scenario

TOLERANCES = [0.0, 0.3, 0.5]
BINS = np.array([0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
ERROR_TOL = 0.05
PROCEDURES = ["uncorrected", "bonferroni", "bh_window", "LORD++"]


def oc_task(args):
    tol, seed, n = args
    onset = 1300
    sc = make_scenario(
        ScenarioConfig(n_streams=n, n_steps=onset + 1000, phi=0.5, rho=0.0, drift_fraction=0.7,
                       magnitude_range=(0.0, 1.0), fixed_onset=onset),
        seed=900 + seed,
    )
    cfg = MonitorConfig(calibration=CalibrationConfig(method="sieve", tolerance=tol))
    t = run_monitor(sc, make_detector("PH"), None, cfg).tests
    post = t[t.t_end >= onset + 500]  # horizon of 5 windows entirely after the shift
    size = sc.mean_shift[post.stream, post.t_end - 1]
    return [{"part": "A", "tolerance": tol, "shift": float(s), "reject": bool(p <= 0.05)}
            for s, p in zip(size, post.pvalue)]


def real_task(args):
    data, tol, seed, n_models = args
    base = insects_scenario(n_models=n_models, seed=seed) if data == "insects" else elec2_scenario(
        n_models=n_models, drift_fraction=0.2, flip=0.5, seed=seed)
    sc = error_rate_view(base, ERROR_TOL)
    cfg = MonitorConfig(calibration=CalibrationConfig(method="moving", tolerance=tol))
    det = make_detector("PH")
    cache, rows = {}, []
    for name in PROCEDURES:
        s = summarize(run_monitor(sc, det, make_procedure(name, 0.05), cfg, seed=seed, cache=cache))
        rows.append({"part": "B", "data": data, "tolerance": tol, "procedure": name, "seed": seed,
                     **{k: s[k] for k in ("alarms", "false_alarms", "fdp", "window_fdp", "power_per_test", "n_tests", "n_null_tests")}})
    return rows


def dispatch(task):
    kind, args = task
    return oc_task(args) if kind == "oc" else real_task(args)


def plot_oc(oc: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    centers = (BINS[:-1] + BINS[1:]) / 2
    for i, tol in enumerate(TOLERANCES):
        d = oc[oc.tolerance == tol]
        idx = np.digitize(d["shift"], BINS) - 1
        rate = [d.reject[idx == b].mean() for b in range(len(centers))]
        ax.plot(centers, rate, label=f"допуск δ = {tol:g}", **line_kwargs(PALETTE[i]))
        if tol > 0:
            ax.axvline(tol, color=PALETTE[i], linewidth=1, alpha=0.6)
    ax.axhline(0.05, color=TEXT_2, linewidth=1)
    ax.set_xlabel("истинный сдвиг ошибки, в стандартных отклонениях")
    ax.set_ylabel("доля тревог при α = 0.05")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left")
    ax.set_title("Кривая отвержения: тест «ошибка выросла больше чем на δ»")
    savefig(fig, "exp8_oc.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    q = args.quick
    n_oc, oc_seeds, n_models, seeds = (100, 1, 12, 1) if q else (600, 3, 50, 3)
    tasks = [("oc", (tol, s, n_oc)) for tol in TOLERANCES for s in range(oc_seeds)]
    tasks += [("real", (d, tol, s, n_models)) for d in ("insects", "elec2") for tol in (0.0, ERROR_TOL) for s in range(seeds)]
    print(f"{len(tasks)} tasks")
    rows = run_parallel(dispatch, tasks)
    oc = rows[rows.part == "A"].dropna(axis=1, how="all")
    real = rows[rows.part == "B"].dropna(axis=1, how="all")
    oc.to_csv(RESULTS / "exp8_oc.csv", index=False)
    real.to_csv(RESULTS / "exp8_real_runs.csv", index=False)
    agg = real.groupby(["data", "tolerance", "procedure"])[["alarms", "false_alarms", "fdp", "window_fdp", "power_per_test"]].mean().reset_index()
    order = {p: i for i, p in enumerate(PROCEDURES)}
    agg = agg.sort_values(["data", "tolerance", "procedure"], key=lambda s: s.map(order) if s.name == "procedure" else s)
    agg.to_csv(RESULTS / "exp8_real.csv", index=False)
    idx = np.digitize(oc["shift"], BINS) - 1
    oc_tab = oc.assign(bin=[f"[{BINS[i]:g}, {BINS[i + 1]:g})" for i in idx]).groupby(["tolerance", "bin"]).reject.agg(["mean", "size"]).reset_index()
    setup_style()
    plot_oc(oc)
    with open(RESULTS / "exp8_tables.md", "w") as f:
        f.write("## A. Доля тревог в зависимости от истинного сдвига (PH, α = 0.05)\n\n" + markdown_table(oc_tab) + "\n\n")
        f.write(f"## B. Реальные данные, истина — рост ошибки больше {ERROR_TOL:g}\n\n")
        f.write("`tolerance` — допуск в калибровке теста (0 — обычный тест «есть ли изменение»).\n\n")
        f.write(markdown_table(agg) + "\n")
    print(oc_tab.round(3).to_string(index=False))
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
