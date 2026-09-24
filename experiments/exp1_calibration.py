"""Experiment 1: do resampled p-values of practical detectors hold their level?

Part A. Stationary AR(1) streams (no drift), phi in {0, 0.5, 0.8}. For every
detector and resampling scheme the empirical false alarm rate of ``p <= alpha``
is compared with the nominal alpha; the uncalibrated river default threshold
is reported alongside.

Part B. A mini-benchmark at a unified false alarm rate: half of the streams
shift at a known step. Detectors are compared at their river defaults (each
at its own, unknown FAR) and after calibration to FAR = 0.05.

Usage: python experiments/exp1_calibration.py [--quick]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import (
    DETECTOR_ORDER,
    N_REF,
    PALETTE,
    RESULTS,
    SURFACE,
    TEXT_2,
    WINDOW,
    default_resampling,
    line_kwargs,
    make_detector,
    markdown_table,
    monitor_config,
    run_parallel,
    savefig,
    setup_style,
)
from driftfdr import ScenarioConfig, make_scenario, run_monitor

ALPHAS = [0.1, 0.05, 0.01, 0.001]
PHIS = [0.0, 0.5, 0.8]
RESAMPLING = ["iid", "moving", "sieve"]
RESAMPLING_LABELS = {"iid": "iid-бутстреп", "moving": "блочный бутстреп", "sieve": "AR-sieve бутстреп"}
RESAMPLING_COLORS = {"moving": PALETTE[0], "sieve": PALETTE[1], "iid": PALETTE[2]}


def null_task(args):
    phi, detector_name, method, n_streams, n_windows = args
    scenario = make_scenario(
        ScenarioConfig(
            n_streams=n_streams,
            n_steps=N_REF + n_windows * WINDOW,
            phi=phi,
            rho=0.0,
            drift_fraction=0.0,
        ),
        seed=100 + int(10 * phi),
    )
    detector = make_detector(detector_name)
    tests = run_monitor(scenario, detector, None, monitor_config(method)).tests
    p = tests["pvalue"].to_numpy()
    row = {
        "phi": phi,
        "detector": detector_name,
        "resampling": method,
        "n_tests": p.size,
        "raw_far": float(np.mean(tests["statistic"] > detector.default_threshold)),
    }
    row.update({f"far@{a:g}": float(np.mean(p <= a)) for a in ALPHAS})
    return [row]


def benchmark_task(args):
    detector_name, magnitude, n_streams = args
    onset = N_REF + 10 * WINDOW
    scenario = make_scenario(
        ScenarioConfig(
            n_streams=n_streams,
            n_steps=onset + 5 * WINDOW,
            phi=0.5,
            rho=0.0,
            drift_fraction=0.5,
            magnitude=magnitude,
            fixed_onset=onset,
        ),
        seed=200,
    )
    detector = make_detector(detector_name)
    tests = run_monitor(scenario, detector, None, monitor_config(default_resampling(detector))).tests
    raw = tests["statistic"] > detector.default_threshold
    cal = tests["pvalue"] <= 0.05
    null = tests["is_null"]
    rows = []
    for lag in (0, 2):
        post = (~null) & (tests["t_end"] == onset + (lag + 1) * WINDOW)
        rows.append(
            {
                "detector": detector_name,
                "magnitude": magnitude,
                "windows_after_onset": lag + 1,
                "raw_far": float(raw[null].mean()),
                "raw_power": float(raw[post].mean()),
                "calibrated_far": float(cal[null].mean()),
                "calibrated_power": float(cal[post].mean()),
            }
        )
    return rows


def plot_calibration(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(PHIS), len(DETECTOR_ORDER), figsize=(12, 8.2), sharex=True, sharey=True)
    for i, phi in enumerate(PHIS):
        for j, det in enumerate(DETECTOR_ORDER):
            ax = axes[i, j]
            ax.plot([1e-3, 0.1], [1e-3, 0.1], color=TEXT_2, linewidth=1, zorder=1)
            for method in RESAMPLING:
                row = df[(df.phi == phi) & (df.detector == det) & (df.resampling == method)]
                if row.empty:
                    continue
                far = [max(row[f"far@{a:g}"].iloc[0], 1e-4) for a in ALPHAS]
                ax.plot(ALPHAS, far, label=RESAMPLING_LABELS[method], **line_kwargs(RESAMPLING_COLORS[method]))
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(8e-5, 1.2)
            if i == 0:
                ax.set_title(det)
            if j == 0:
                ax.set_ylabel(f"φ = {phi:g}\nфактический FAR")
            if i == len(PHIS) - 1:
                ax.set_xlabel("номинальный уровень α")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(
        "Доля ложных тревог калиброванных p-значений на стационарных потоках (серая линия: FAR = α)",
        y=1.07,
        fontsize=12,
        fontweight="bold",
    )
    fig.text(0.5, -0.01, "DDM работает на бинарном сигнале ошибок, AR-sieve для него неприменим", ha="center", color=TEXT_2, fontsize=8.5)
    savefig(fig, "exp1_calibration.png")


def plot_benchmark(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    sub = df[(df.windows_after_onset == 1) & (df.magnitude == df.magnitude.max())]
    sub = sub.set_index("detector").loc[DETECTOR_ORDER]
    x = np.arange(len(DETECTOR_ORDER))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    panels = [
        ("far", "Доля ложных тревог на окно", "FAR"),
        ("power", f"Мощность в первом окне после сдвига (Δ = {sub.magnitude.iloc[0]:g})", "доля обнаружений"),
    ]
    series = [("raw", "пороги river по умолчанию", PALETTE[1]), ("calibrated", "калибровка, FAR = 0.05", PALETTE[0])]
    for ax, (metric, title, ylabel) in zip(axes, panels):
        for k, (prefix, label, color) in enumerate(series):
            vals = sub[f"{prefix}_{metric}"].to_numpy()
            bars = ax.bar(x + (k - 0.5) * (width + 0.02), vals, width, color=color, label=label, edgecolor=SURFACE, linewidth=2)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=TEXT_2)
        ax.set_xticks(x, DETECTOR_ORDER)
        ax.set_ylim(0, 1.1)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
    axes[0].axhline(0.05, color=TEXT_2, linewidth=1)
    axes[0].legend(loc="upper left")
    fig.suptitle("Сравнение детекторов при своих порогах и при едином FAR (φ = 0.5)", y=1.04, fontsize=12, fontweight="bold")
    savefig(fig, "exp1_benchmark.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small run for a smoke test")
    args = parser.parse_args()
    n_streams, n_windows = (100, 10) if args.quick else (1000, 20)

    tasks = [
        (phi, det, method, n_streams, n_windows)
        for phi in PHIS
        for det in DETECTOR_ORDER
        for method in RESAMPLING
        if not (method == "sieve" and make_detector(det).input_kind == "errors")
    ]
    print(f"Part A: {len(tasks)} cells")
    calib = run_parallel(null_task, tasks).sort_values(["phi", "detector", "resampling"])
    calib.to_csv(RESULTS / "exp1_calibration.csv", index=False)

    tasks = [(det, mag, n_streams) for det in DETECTOR_ORDER for mag in (0.5, 1.0)]
    print(f"Part B: {len(tasks)} cells")
    bench = run_parallel(benchmark_task, tasks).sort_values(["magnitude", "windows_after_onset", "detector"])
    bench.to_csv(RESULTS / "exp1_benchmark.csv", index=False)

    setup_style()
    plot_calibration(calib)
    plot_benchmark(bench)

    cols = ["phi", "detector", "resampling", "raw_far"] + [f"far@{a:g}" for a in ALPHAS]
    with open(RESULTS / "exp1_tables.md", "w") as f:
        f.write("## Калибровка на стационарных потоках\n\n")
        f.write(markdown_table(calib[cols]) + "\n\n")
        f.write("## Бенчмарк при едином FAR\n\n")
        f.write(markdown_table(bench) + "\n")
    print(calib[cols].round(4).to_string(index=False))
    print(bench.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
