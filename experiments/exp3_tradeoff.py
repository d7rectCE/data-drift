"""Experiment 3: the trade-off curve between false retrains and detection delay.

For each decision rule the level alpha is swept, tracing a curve of
(false retrains per 1000 stream-steps, steps a drifted model keeps running).
The question behind the applied hypothesis is whether multiplicity-aware
rules lie below the per-stream rule at the same false-retrain budget, i.e.
whether they do better than simply lowering every stream's threshold, and
how this depends on the number of streams K.

Usage: python experiments/exp3_tradeoff.py [--quick]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import (
    PROCEDURE_COLORS,
    PROCEDURE_LABELS,
    RESULTS,
    line_kwargs,
    markdown_table,
    run_parallel,
    run_procedures,
    savefig,
    setup_style,
)
from driftfdr import ScenarioConfig

PROCEDURES = ["LORD++", "uncorrected", "bh_window", "LOND", "bonferroni"]
ALPHAS = [1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2]


def task(args):
    n_streams, seed = args
    config = ScenarioConfig(n_streams=n_streams, n_steps=5000, phi=0.5, rho=0.3, drift_fraction=0.1, magnitude=1.0)
    procs = [(p, a) for p in PROCEDURES for a in ALPHAS]
    rows = run_procedures(config, seed, "PH", procs)
    return [{"n_streams": n_streams, **r} for r in rows]


def plot(agg: pd.DataFrame):
    import matplotlib.pyplot as plt

    ks = sorted(agg.n_streams.unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(4.2 * len(ks), 4), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), ks):
        for proc in PROCEDURES:
            d = agg[(agg.n_streams == k) & (agg.procedure == proc)].sort_values("alpha")
            ax.plot(d.false_alarms_per_1k_stream_steps, d.degraded_per_drift, label=PROCEDURE_LABELS[proc], **line_kwargs(PROCEDURE_COLORS[proc]))
            at = d[d.alpha == 0.05]
            ax.plot(at.false_alarms_per_1k_stream_steps, at.degraded_per_drift, "o",
                    markersize=11, markerfacecolor="none", markeredgecolor=PROCEDURE_COLORS[proc], markeredgewidth=1.5)
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlim(left=-0.001)
        ax.set_title(f"K = {k}")
        ax.set_xlabel("ложных переобучений на 1000 шагов потока")
    first = np.atleast_1d(axes)[0]
    first.set_ylabel("шагов на старой модели после дрейфа")
    handles, labels = first.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(PROCEDURES), bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(
        "Компромисс «ложные переобучения ↔ задержка»: кривые по уровню α, кружок = α 0.05",
        y=1.13,
        fontsize=12,
        fontweight="bold",
    )
    fig.text(0.5, -0.04, "Ниже и левее — лучше; ось X линейна до 0.01 и логарифмична дальше. Page-Hinkley, 10% потоков с дрейфом, φ = 0.5, ρ = 0.3",
             ha="center", fontsize=8.5, color="#52514e")
    savefig(fig, "exp3_tradeoff.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    ks, seeds = ([10, 50], range(2)) if args.quick else ([10, 100, 500], range(6))
    tasks = [(k, s) for k in sorted(ks, reverse=True) for s in seeds]
    print(f"{len(tasks)} tasks")
    raw = run_parallel(task, tasks)
    raw.to_csv(RESULTS / "exp3_runs.csv", index=False)
    cols = ["false_alarms_per_1k_stream_steps", "degraded_per_drift", "fdp", "mdr", "mean_delay"]
    agg = raw.groupby(["n_streams", "procedure", "alpha"])[cols].mean().reset_index()
    agg.to_csv(RESULTS / "exp3_tradeoff.csv", index=False)
    setup_style()
    plot(agg)
    with open(RESULTS / "exp3_tables.md", "w") as f:
        f.write("## Кривые компромисса (среднее по сценариям)\n\n")
        f.write(markdown_table(agg.rename(columns={"fdp": "fdr"})) + "\n")
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
