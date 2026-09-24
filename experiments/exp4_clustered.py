"""Experiment 4 (stage 2): does FDR adaptivity pay off when drifts come in clusters?

Stage 1 found that, with rare sporadic drifts, FDR rules are no better than a
per-stream threshold scaled with K. Adaptive rules can only gain when many
streams change in the same window, so here the same 20% of streams drift
either sporadically, in two events of 10% each, or in one event of 20%.
Each rule's level is swept to trace its trade-off curve, and rules are
compared by the best delay they reach within a fixed false-retrain budget.

Usage: python experiments/exp4_clustered.py [--quick]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import (
    PROCEDURE_COLORS,
    PROCEDURE_LABELS,
    RESULTS,
    SURFACE,
    TEXT_2,
    line_kwargs,
    markdown_table,
    run_parallel,
    run_procedures,
    savefig,
    setup_style,
)
from driftfdr import ScenarioConfig

PROCEDURES = ["LORD++", "uncorrected", "bh_window", "storey_bh", "BatchBH", "bonferroni"]
ALPHAS = [1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2]
SCENARIOS = {
    "sporadic": dict(drift_fraction=0.2),
    "2 events": dict(drift_fraction=0.0, drift_events=2, event_fraction=0.1),
    "1 event": dict(drift_fraction=0.0, drift_events=1, event_fraction=0.2),
}
SCENARIO_LABELS = {
    "sporadic": "разрозненные дрейфы",
    "2 events": "2 события по 10% потоков",
    "1 event": "1 событие, 20% потоков",
}
BUDGETS = [0.001, 0.01]
METRICS = ["false_alarms_per_1k_stream_steps", "degraded_per_drift", "fdp", "window_fdp", "mdr", "mean_delay"]


def task(args):
    n_streams, scenario, seed = args
    config = ScenarioConfig(n_streams=n_streams, n_steps=5000, phi=0.5, rho=0.3, magnitude=1.0, **SCENARIOS[scenario])
    procs = [(p, a) for p in PROCEDURES for a in ALPHAS]
    rows = run_procedures(config, seed, "PH", procs)
    return [{"n_streams": n_streams, "scenario": scenario, **r} for r in rows]


def best_within_budget(agg: pd.DataFrame) -> pd.DataFrame:
    """Smallest mean degraded steps per drift over alphas whose false-retrain rate fits the budget."""
    rows = []
    for (k, sc, proc), g in agg.groupby(["n_streams", "scenario", "procedure"]):
        for budget in BUDGETS:
            ok = g[g.false_alarms_per_1k_stream_steps <= budget]
            best = ok.loc[ok.degraded_per_drift.idxmin()] if not ok.empty else None
            rows.append(
                {
                    "n_streams": k,
                    "scenario": sc,
                    "procedure": proc,
                    "budget": budget,
                    "degraded_per_drift": np.nan if best is None else best.degraded_per_drift,
                    "alpha": np.nan if best is None else best.alpha,
                }
            )
    out = pd.DataFrame(rows)
    base = out[out.procedure == "uncorrected"].set_index(["n_streams", "scenario", "budget"]).degraded_per_drift
    out["vs_fixed_threshold"] = out.degraded_per_drift / base.reindex(
        pd.MultiIndex.from_frame(out[["n_streams", "scenario", "budget"]])
    ).to_numpy()
    return out


def plot_curves(agg: pd.DataFrame):
    import matplotlib.pyplot as plt

    ks = sorted(agg.n_streams.unique())
    scenarios = [s for s in SCENARIOS if s in set(agg.scenario)]
    fig, axes = plt.subplots(len(ks), len(scenarios), figsize=(4.3 * len(scenarios), 3.8 * len(ks)), sharey="row", squeeze=False)
    for i, k in enumerate(ks):
        for j, sc in enumerate(scenarios):
            ax = axes[i, j]
            for proc in PROCEDURES:
                d = agg[(agg.n_streams == k) & (agg.scenario == sc) & (agg.procedure == proc)].sort_values("alpha")
                ax.plot(d.false_alarms_per_1k_stream_steps, d.degraded_per_drift, label=PROCEDURE_LABELS[proc], **line_kwargs(PROCEDURE_COLORS[proc]))
            ax.set_xscale("symlog", linthresh=0.001)
            ax.set_xlim(left=-0.0002)
            ax.set_ylim(bottom=0, top=min(ax.get_ylim()[1], 1500))
            ax.set_title(f"{SCENARIO_LABELS[sc]}, K = {k}")
            if i == len(ks) - 1:
                ax.set_xlabel("ложных переобучений на 1000 шагов потока")
            if j == 0:
                ax.set_ylabel("шагов на старой модели после дрейфа")
    fig.tight_layout(rect=(0, 0.02, 1, 0.9))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(PROCEDURES), bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Кривые компромисса при разрозненных и кластерных дрейфах (ниже и левее — лучше)", y=0.995, fontsize=12, fontweight="bold")
    fig.text(0.5, 0.005, "Page-Hinkley, 20% потоков с дрейфом, φ = 0.5, ρ = 0.3; ось X линейна до 0.001", ha="center", fontsize=8.5, color=TEXT_2)
    savefig(fig, "exp4_clustered_curves.png")


def plot_budget(best: pd.DataFrame, budget: float):
    import matplotlib.pyplot as plt

    sub = best[best.budget == budget]
    ks = sorted(sub.n_streams.unique())
    scenarios = [s for s in SCENARIOS if s in set(sub.scenario)]
    fig, axes = plt.subplots(1, len(ks), figsize=(6 * len(ks), 3.8), sharey=True, squeeze=False)
    width = 0.8 / len(PROCEDURES)
    x = np.arange(len(scenarios))
    for ax, k in zip(axes[0], ks):
        for i, proc in enumerate(PROCEDURES):
            vals = [sub[(sub.n_streams == k) & (sub.scenario == sc) & (sub.procedure == proc)].degraded_per_drift.iloc[0] for sc in scenarios]
            ax.bar(x + (i - (len(PROCEDURES) - 1) / 2) * width, vals, width * 0.92, color=PROCEDURE_COLORS[proc], label=PROCEDURE_LABELS[proc], edgecolor=SURFACE, linewidth=1)
        ax.set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios])
        ax.set_title(f"K = {k}")
        ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("шагов на старой модели после дрейфа")
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(PROCEDURES), bbox_to_anchor=(0.5, 0.93))
    fig.suptitle(f"Лучшая задержка при бюджете ≤ {budget:g} ложных переобучений на 1000 шагов потока (ниже — лучше)", y=0.995, fontsize=12, fontweight="bold")
    savefig(fig, f"exp4_budget_{budget:g}.png")


def replot():
    setup_style()
    agg = pd.read_csv(RESULTS / "exp4_tradeoff.csv")
    best = pd.read_csv(RESULTS / "exp4_budget.csv")
    plot_curves(agg)
    for b in BUDGETS:
        plot_budget(best, b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    ks, seeds = ([50], range(2)) if args.quick else ([100, 500], range(6))
    tasks = [(k, sc, s) for k in sorted(ks, reverse=True) for sc in SCENARIOS for s in seeds]
    print(f"{len(tasks)} tasks")
    raw = run_parallel(task, tasks)
    raw.to_csv(RESULTS / "exp4_runs.csv", index=False)
    agg = raw.groupby(["n_streams", "scenario", "procedure", "alpha"])[METRICS].mean().reset_index()
    agg.to_csv(RESULTS / "exp4_tradeoff.csv", index=False)
    best = best_within_budget(agg)
    best.to_csv(RESULTS / "exp4_budget.csv", index=False)
    replot()
    at05 = agg[agg.alpha == 0.05].drop(columns="alpha")
    with open(RESULTS / "exp4_tables.md", "w") as f:
        f.write("## Лучшая задержка в пределах бюджета ложных переобучений\n\n")
        f.write("`vs_fixed_threshold` — отношение к лучшему порогу на поток без поправки при том же бюджете.\n\n")
        f.write(markdown_table(best) + "\n\n")
        f.write("## Все правила при α = 0.05\n\n" + markdown_table(at05.rename(columns={"fdp": "fdr"})) + "\n")
    print(best.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
