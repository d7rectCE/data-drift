"""Experiment 6 (stage 2): repairing the deep tail of the calibration.

Experiments 1 and 5 showed that resampled p-values are too small at the
levels online corrections use (Bonferroni gives a per-window FWER of about
0.1 instead of 0.05 even for independent streams). The suspected cause is that
the bootstrap treats the model fitted to 300 reference points as exact. The
``sieve_pu`` scheme redraws the AR coefficients for every replicate (see
``bootstrap.py``). Here it is compared with the plain sieve on:

A. the false alarm rate of ``p <= alpha`` down to alpha = 1e-4;
B. the per-window FWER of Bonferroni at alpha / K, the direct test of the tail;
C. the detection power it costs, after an abrupt shift.

Usage: python experiments/exp6_tail.py [--quick]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import (
    N_REF,
    PALETTE,
    RESULTS,
    SURFACE,
    TEXT_2,
    WINDOW,
    line_kwargs,
    make_detector,
    markdown_table,
    monitor_config,
    run_parallel,
    savefig,
    setup_style,
)
from driftfdr import ScenarioConfig, make_scenario, run_monitor

METHODS = ["sieve", "sieve_pu"]
METHOD_LABELS = {"sieve": "AR-sieve", "sieve_pu": "AR-sieve + неопределённость параметров"}
METHOD_COLORS = {"sieve": PALETTE[1], "sieve_pu": PALETTE[0]}
DETECTORS = ["PH", "ADWIN", "KS"]
ALPHAS = [0.05, 0.01, 1e-3, 1e-4]


def far_task(args):
    phi, det, method, n_streams, n_windows = args
    sc = make_scenario(
        ScenarioConfig(n_streams=n_streams, n_steps=N_REF + n_windows * WINDOW, phi=phi, rho=0.0, drift_fraction=0.0),
        seed=600 + int(10 * phi),
    )
    p = run_monitor(sc, make_detector(det), None, monitor_config(method)).tests["pvalue"].to_numpy()
    row = {"part": "A", "phi": phi, "detector": det, "method": method, "n_tests": p.size}
    row.update({f"far@{a:g}": float(np.mean(p <= a)) for a in ALPHAS})
    return [row]


def fwer_task(args):
    det, method, n_streams, seed = args
    sc = make_scenario(ScenarioConfig(n_streams=n_streams, n_steps=5000, phi=0.5, rho=0.0, drift_fraction=0.0), seed=700 + seed)
    tests = run_monitor(sc, make_detector(det), None, monitor_config(method)).tests
    min_p = tests.groupby("window")["pvalue"].min().to_numpy()
    return [{"part": "B", "detector": det, "method": method, "seed": seed, "n_streams": n_streams,
             "fwer": float(np.mean(min_p <= 0.05 / n_streams)), "n_windows": min_p.size}]


def power_task(args):
    det, method, magnitude, n_streams = args
    onset = N_REF + 10 * WINDOW
    sc = make_scenario(
        ScenarioConfig(n_streams=n_streams, n_steps=onset + 5 * WINDOW, phi=0.5, rho=0.0, drift_fraction=0.5,
                       magnitude=magnitude, fixed_onset=onset),
        seed=800,
    )
    t = run_monitor(sc, make_detector(det), None, monitor_config(method)).tests
    rows = []
    for lag in (1, 3):
        post = (~t.is_null) & (t.t_end == onset + lag * WINDOW)
        for a in (0.05, 1e-3):
            rows.append({"part": "C", "detector": det, "method": method, "magnitude": magnitude,
                         "windows_after_onset": lag, "alpha": a, "power": float((t.pvalue[post] <= a).mean())})
    return rows


def plot(far: pd.DataFrame, fwer: pd.DataFrame):
    import matplotlib.pyplot as plt

    phis = sorted(far.phi.unique())
    fig, axes = plt.subplots(1, len(DETECTORS) + 1, figsize=(15, 3.9))
    for ax, det in zip(axes, DETECTORS):
        ax.plot([1e-4, 0.05], [1e-4, 0.05], color=TEXT_2, linewidth=1)
        for method in METHODS:
            for phi, marker in zip(phis, ("o", "s")):
                r = far[(far.detector == det) & (far.method == method) & (far.phi == phi)]
                y = [max(r[f"far@{a:g}"].iloc[0], 3e-5) for a in ALPHAS]
                kw = line_kwargs(METHOD_COLORS[method])
                kw["marker"] = marker
                ax.plot(ALPHAS, y, label=f"{METHOD_LABELS[method]}, φ = {phi:g}", **kw)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(2e-5, 0.3)
        ax.set_title(det)
        ax.set_xlabel("номинальный уровень α")
    axes[0].set_ylabel("фактический FAR")
    ax = axes[-1]
    x = np.arange(len(DETECTORS))
    width = 0.36
    for i, method in enumerate(METHODS):
        g = fwer[fwer.method == method].groupby("detector").fwer
        mean, se = g.mean().reindex(DETECTORS), (g.std() / np.sqrt(g.size())).reindex(DETECTORS)
        ax.bar(x + (i - 0.5) * (width + 0.02), mean, width, yerr=se, color=METHOD_COLORS[method], edgecolor=SURFACE,
               linewidth=2, error_kw={"ecolor": TEXT_2, "elinewidth": 1, "capsize": 0})
    ax.axhline(0.05, color=TEXT_2, linewidth=1)
    ax.set_xticks(x, DETECTORS)
    ax.grid(axis="x", visible=False)
    ax.set_title("Бонферрони: P(ложная тревога в окне)")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("Калибровка глубокого хвоста: серая линия — номинальный уровень", y=0.995, fontsize=12, fontweight="bold")
    savefig(fig, "exp6_tail.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    q = args.quick
    n_far, n_win, n_fwer, n_seeds = (100, 10, 50, 2) if q else (1000, 20, 100, 40)
    tasks = [("far", (phi, d, m, n_far, n_win)) for phi in (0.5, 0.8) for d in DETECTORS for m in METHODS]
    tasks += [("fwer", (d, m, n_fwer, s)) for d in DETECTORS for m in METHODS for s in range(n_seeds)]
    tasks += [("power", (d, m, mag, 200 if q else 1000)) for d in DETECTORS for m in METHODS for mag in (0.5, 1.0)]
    print(f"{len(tasks)} tasks")
    rows = run_parallel(dispatch, tasks)
    far, fwer, power = (rows[rows.part == k].dropna(axis=1, how="all") for k in "ABC")
    far.to_csv(RESULTS / "exp6_far.csv", index=False)
    fwer.to_csv(RESULTS / "exp6_fwer_runs.csv", index=False)
    power.to_csv(RESULTS / "exp6_power.csv", index=False)
    setup_style()
    plot(far, fwer)
    g = fwer.groupby(["detector", "method"]).fwer
    fwer_sum = pd.DataFrame({"fwer": g.mean(), "se": g.std() / np.sqrt(g.size()), "n_scenarios": g.size()}).reset_index()
    pw = power.pivot_table(index=["detector", "magnitude", "windows_after_onset", "alpha"], columns="method", values="power").reset_index()
    with open(RESULTS / "exp6_tables.md", "w") as f:
        f.write("## A. Фактический FAR\n\n" + markdown_table(far.drop(columns="part").sort_values(["phi", "detector", "method"])) + "\n\n")
        f.write(f"## B. Бонферрони, P(ложная тревога в окне), K = {n_fwer}, ρ = 0\n\n" + markdown_table(fwer_sum) + "\n\n")
        f.write("## C. Мощность\n\n" + markdown_table(pw) + "\n")
    print(far.drop(columns="part").round(5).to_string(index=False))
    print(fwer_sum.round(4).to_string(index=False))
    print(pw.round(3).to_string(index=False))


def dispatch(task):
    kind, args = task
    return {"far": far_task, "fwer": fwer_task, "power": power_task}[kind](args)


if __name__ == "__main__":
    main()
