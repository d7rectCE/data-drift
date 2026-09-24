"""Experiment 5 (stage 2): how much can a dependence-aware correction gain?

The proposal expects naive corrections to become overly conservative when
streams are correlated. Before building a joint resampling procedure, this
measures the ceiling of its benefit on drift-free streams with cross-stream
correlation rho:

* the per-window FWER of Bonferroni (alpha / K per stream);
* the oracle per-stream level that gives exactly alpha per-window FWER under
  the actual dependence (the 5% quantile of the per-window minimum p-value),
  and the implied effective number of tests ``M_eff = alpha / level``;
* the dispersion index (variance / mean) of the number of false alarms per
  window at a fixed per-stream level, which is 1 for independent streams and
  grows when false alarms come in bursts.

With strong correlation all streams of one scenario move together, so the
effective sample size is the number of scenarios, not of windows; intervals
are therefore obtained by resampling whole scenarios.

Usage: python experiments/exp5_dependence.py [--quick]
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
from driftfdr import ScenarioConfig, make_scenario, run_monitor

ALPHA = 0.05
BURST_LEVEL = 0.01
RHOS = [0.0, 0.3, 0.6, 0.9]


def task(args):
    n_streams, rho, seed = args
    scenario = make_scenario(
        ScenarioConfig(n_streams=n_streams, n_steps=5000, phi=0.5, rho=rho, drift_fraction=0.0), seed=seed
    )
    tests = run_monitor(scenario, make_detector("PH"), None, monitor_config("sieve")).tests
    g = tests.groupby("window")["pvalue"]
    per_window = pd.DataFrame(
        {"min_p": g.min(), "n_below": g.apply(lambda p: int((p <= BURST_LEVEL).sum()))}
    ).reset_index()
    per_window["n_streams"], per_window["rho"], per_window["seed"] = n_streams, rho, seed
    return per_window.to_dict("records")


def _stats(g: pd.DataFrame, k: int) -> dict:
    oracle = float(np.quantile(g.min_p, ALPHA))
    return {
        "bonferroni_fwer": float(np.mean(g.min_p <= ALPHA / k)),
        "oracle_level": oracle,
        "m_eff": ALPHA / max(oracle, 1e-300),
        "mean_fa_at_0.01": float(g.n_below.mean()),
        "dispersion_index": float(g.n_below.var() / max(g.n_below.mean(), 1e-12)),
    }


def summarize(windows: pd.DataFrame, n_resamples: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for (k, rho), g in windows.groupby(["n_streams", "rho"]):
        row = {"n_streams": k, "rho": rho, "n_scenarios": g.seed.nunique(), "n_windows": len(g), **_stats(g, k)}
        by_seed = {s: d for s, d in g.groupby("seed")}
        seeds = np.array(list(by_seed))
        boot = [
            _stats(pd.concat([by_seed[s] for s in rng.choice(seeds, seeds.size)]), k)
            for _ in range(n_resamples)
        ]
        for key in ("bonferroni_fwer", "m_eff", "dispersion_index"):
            lo, hi = np.quantile([b[key] for b in boot], [0.05, 0.95])
            row[f"{key}_lo"], row[f"{key}_hi"] = float(lo), float(hi)
        rows.append(row)
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    panels = [
        ("bonferroni_fwer", "P(ложная тревога в окне) у Бонферрони", "linear"),
        ("m_eff", "Эффективное число тестов / K", "log"),
        ("dispersion_index", f"Кучкование ложных тревог (p ≤ {BURST_LEVEL:g})", "log"),
    ]
    for i, k in enumerate(sorted(summary.n_streams.unique())):
        d = summary[summary.n_streams == k].sort_values("rho")
        for ax, (key, title, scale) in zip(axes, panels):
            norm = k if key == "m_eff" else 1
            ax.fill_between(d.rho, d[f"{key}_lo"] / norm, d[f"{key}_hi"] / norm, color=PALETTE[i], alpha=0.1, linewidth=0)
            ax.plot(d.rho, d[key] / norm, label=f"K = {k}", **line_kwargs(PALETTE[i]))
    for ax, (key, title, scale) in zip(axes, panels):
        ax.set_title(title)
        ax.set_yscale(scale)
        ax.set_xlabel("корреляция между потоками ρ")
        ax.set_xticks(RHOS)
    axes[0].axhline(ALPHA, color=TEXT_2, linewidth=1)
    axes[1].axhline(1, color=TEXT_2, linewidth=1)
    axes[2].axhline(1, color=TEXT_2, linewidth=1)
    axes[2].set_ylabel("дисперсия / среднее")
    axes[0].legend(loc="upper left")
    fig.suptitle(
        "Зависимость между потоками без дрейфа: Page-Hinkley, φ = 0.5, заливка — 90% интервал по сценариям",
        y=1.03,
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    savefig(fig, "exp5_dependence.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n_seeds = {50: 3} if args.quick else {500: 16, 100: 60}
    tasks = [(k, rho, s) for k, n in n_seeds.items() for rho in RHOS for s in range(n)]
    print(f"{len(tasks)} tasks")
    windows = run_parallel(task, tasks)
    windows.to_csv(RESULTS / "exp5_windows.csv", index=False)
    summary = summarize(windows)
    summary.to_csv(RESULTS / "exp5_dependence.csv", index=False)
    setup_style()
    plot(summary)
    with open(RESULTS / "exp5_tables.md", "w") as f:
        f.write("## Потолок выигрыша от учёта зависимости\n\n" + markdown_table(summary) + "\n")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
