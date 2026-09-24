"""Experiment 2: what happens to false alarms and delays as the number of streams grows.

K streams, 10% of which undergo an abrupt mean shift of one standard
deviation at a random time; AR(1) with phi = 0.5, common-factor correlation
rho in {0, 0.3}. Page-Hinkley with AR-sieve calibration is run under every
decision rule; results are averaged over independent scenarios (FDR is the
mean FDP). A second part compares detectors at K = 100.

Usage: python experiments/exp2_scaling.py [--quick]
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

PROCEDURES = ["uncorrected", "bonferroni", "bh_window", "LOND", "LORD++", "SAFFRON", "alpha-investing"]
PLOTTED = ["LORD++", "uncorrected", "bh_window", "SAFFRON", "LOND", "bonferroni", "alpha-investing"]
ALPHA = 0.05
METRICS = [
    "fdp",
    "false_alarms_per_window",
    "p_any_false_alarm_per_window",
    "false_alarms_per_1k_stream_steps",
    "mdr",
    "mean_delay",
    "degraded_per_drift",
    "alarms",
    "false_alarms",
    "n_drifts",
]


def scenario_config(n_streams, rho):
    return ScenarioConfig(n_streams=n_streams, n_steps=5000, phi=0.5, rho=rho, drift_fraction=0.1, magnitude=1.0)


def task(args):
    detector, n_streams, rho, seed = args
    procs = [("raw", np.nan)] + [(p, ALPHA) for p in PROCEDURES]
    rows = run_procedures(scenario_config(n_streams, rho), seed, detector, procs)
    return [{"detector": detector, "n_streams": n_streams, "rho": rho, **r} for r in rows]


def aggregate(df: pd.DataFrame, keys) -> pd.DataFrame:
    g = df.groupby(keys)
    out = g[METRICS].mean()
    out["fdr_se"] = g["fdp"].std() / np.sqrt(g.size())
    out["n_seeds"] = g.size()
    return out.reset_index().rename(columns={"fdp": "fdr"})


def breakeven(agg: pd.DataFrame) -> pd.DataFrame:
    """Degraded steps paid per false retrain saved, relative to the uncorrected rule.

    A procedure pays off when one false retrain costs more than this many
    steps of running a drifted model; 0 means it is not slower at all (a false
    alarm also pauses monitoring while a new reference is collected).
    """
    rows = []
    for (det, k, rho), g in agg.groupby(["detector", "n_streams", "rho"]):
        base = g[g.procedure == "uncorrected"].iloc[0]
        for _, r in g[~g.procedure.isin(["uncorrected", "raw"])].iterrows():
            saved = (base.false_alarms - r.false_alarms) / base.n_drifts
            extra = r.degraded_per_drift - base.degraded_per_drift
            rows.append(
                {
                    "detector": det,
                    "n_streams": k,
                    "rho": rho,
                    "procedure": r.procedure,
                    "saved_false_retrains_per_drift": saved,
                    "extra_degraded_steps_per_drift": extra,
                    "breakeven_steps_per_false_retrain": max(extra, 0.0) / saved if saved > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_scaling(agg: pd.DataFrame, rho: float):
    import matplotlib.pyplot as plt

    sub = agg[(agg.detector == "PH") & (agg.rho == rho)]
    panels = [
        ("fdr", "FDR (средняя доля ложных среди тревог)", "linear"),
        ("false_alarms_per_window", "Ложных тревог на окно по системе", "symlog"),
        ("p_any_false_alarm_per_window", "P(хотя бы одна ложная тревога в окне)", "linear"),
        ("degraded_per_drift", "Шагов работы на старой модели после дрейфа", "linear"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    for ax, (metric, title, scale) in zip(axes.flat, panels):
        for proc in PLOTTED:
            d = sub[sub.procedure == proc].sort_values("n_streams")
            ax.plot(d.n_streams, d[metric], label=PROCEDURE_LABELS[proc], **line_kwargs(PROCEDURE_COLORS[proc]))
        ax.set_xscale("log")
        ax.set_xticks(sorted(sub.n_streams.unique()), [str(k) for k in sorted(sub.n_streams.unique())])
        ax.minorticks_off()
        if scale == "symlog":
            ax.set_yscale("symlog", linthresh=0.01)
            ax.set_ylim(bottom=0)
        ax.set_title(title)
        ax.set_xlabel("число потоков K")
    axes[0, 0].axhline(ALPHA, color=TEXT_2, linewidth=1)
    axes[0, 0].set_ylim(0, 1)
    axes[1, 0].axhline(ALPHA, color=TEXT_2, linewidth=1)
    axes[1, 0].set_ylim(0, 1)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(
        f"Масштабирование: Page-Hinkley, 10% потоков с дрейфом, φ = 0.5, ρ = {rho:g}, α = {ALPHA:g}",
        y=0.995,
        fontsize=12,
        fontweight="bold",
    )
    savefig(fig, f"exp2_scaling_rho{rho:g}.png")


def plot_dependence(agg: pd.DataFrame):
    import matplotlib.pyplot as plt

    ph = agg[agg.detector == "PH"]
    k = ph.n_streams.max()
    sub = ph[ph.n_streams == k].set_index(["procedure", "rho"])
    rhos = sorted(agg.rho.unique())
    x = np.arange(len(PLOTTED))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 3.6))
    shades = {rhos[0]: "#86b6ef", rhos[-1]: "#1c5cab"}  # ordinal steps of one hue
    for i, rho in enumerate(rhos):
        vals = np.array([sub.loc[(p, rho), "fdr"] for p in PLOTTED])
        err = np.array([sub.loc[(p, rho), "fdr_se"] for p in PLOTTED])
        ax.bar(x + (i - 0.5) * (width + 0.02), vals, width, yerr=err, color=shades[rho], label=f"ρ = {rho:g}",
               edgecolor=SURFACE, linewidth=2, error_kw={"ecolor": TEXT_2, "elinewidth": 1, "capsize": 0})
    ax.axhline(ALPHA, color=TEXT_2, linewidth=1)
    ax.set_xticks(x, [PROCEDURE_LABELS[p] for p in PLOTTED], rotation=15)
    ax.set_ylabel("FDR")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ax.set_title(f"Зависимость между потоками и FDR, K = {k}")
    savefig(fig, "exp2_dependence.png")


def replot():
    setup_style()
    agg = pd.read_csv(RESULTS / "exp2_scaling.csv")
    for rho in sorted(agg.rho.unique()):
        plot_scaling(agg, rho)
    plot_dependence(agg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        ks, seeds, rhos, det_seeds = [10, 50], range(2), [0.0, 0.3], range(1)
    else:
        ks, seeds, rhos, det_seeds = [10, 50, 100, 250, 500], range(10), [0.0, 0.3], range(5)

    tasks = [("PH", k, rho, s) for k in sorted(ks, reverse=True) for rho in rhos for s in seeds]
    tasks += [(det, 100, 0.3, s) for det in ("ADWIN", "KS", "DDM") for s in det_seeds]
    print(f"{len(tasks)} tasks")
    raw = run_parallel(task, tasks)
    raw.to_csv(RESULTS / "exp2_runs.csv", index=False)
    agg = aggregate(raw, ["detector", "n_streams", "rho", "procedure"])
    agg.to_csv(RESULTS / "exp2_scaling.csv", index=False)
    be = breakeven(agg)
    be.to_csv(RESULTS / "exp2_breakeven.csv", index=False)

    setup_style()
    for rho in rhos:
        plot_scaling(agg, rho)
    plot_dependence(agg)

    cols = ["n_streams", "procedure", "fdr", "false_alarms_per_window", "p_any_false_alarm_per_window", "mdr", "mean_delay", "degraded_per_drift"]
    order = {p: i for i, p in enumerate(["raw"] + PROCEDURES)}
    with open(RESULTS / "exp2_tables.md", "w") as f:
        for rho in rhos:
            t = agg[(agg.detector == "PH") & (agg.rho == rho)].sort_values(["n_streams", "procedure"], key=lambda s: s.map(order) if s.name == "procedure" else s)
            f.write(f"## Page-Hinkley, ρ = {rho:g}\n\n{markdown_table(t[cols])}\n\n")
        det_k = agg[(agg.n_streams == 100) & (agg.rho == 0.3)].sort_values(["detector", "procedure"], key=lambda s: s.map(order) if s.name == "procedure" else s)
        f.write(f"## Детекторы при K = 100, ρ = 0.3\n\n{markdown_table(det_k[['detector'] + cols])}\n\n")
        b = be[(be.detector == "PH") & (be.rho == 0.3)].sort_values(["procedure", "n_streams"])
        f.write(f"## Точка безубыточности относительно правила без поправки (PH, ρ = 0.3)\n\n{markdown_table(b.drop(columns=['detector', 'rho']))}\n")
    print(agg[agg.detector == "PH"][["n_streams", "rho", "procedure", "fdr", "false_alarms_per_window", "mdr", "degraded_per_drift"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
