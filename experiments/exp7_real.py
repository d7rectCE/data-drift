"""Experiment 7 (stage 2): transfer to real data.

A fleet of 50 models (logistic regressions on random feature subsets) is
monitored on two real data sets kept in their original time order.

INSECTS: documented concept drifts shared by all models, plus long runs of a
single class that the documented change points do not mark. Results are
reported under both the documented and the extended ground truth.

Electricity: no documented drift. Label-flip drift (half of the labels flipped, i.e. the concept is lost) is injected into 20% of
the models, either at individual times or in two simultaneous events; the
remaining models keep the natural seasonality of the data, which the regime
null counts as "no change".

Part A checks calibration on the stable first INSECTS concept; part B runs the
decision rules.

Usage: python experiments/exp7_real.py [--quick] [--method sieve|sieve_pu]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import (
    N_REF,
    PALETTE,
    PROCEDURE_COLORS,
    PROCEDURE_LABELS,
    RESULTS,
    SURFACE,
    TEXT_2,
    WINDOW,
    make_detector,
    markdown_table,
    monitor_config,
    run_parallel,
    savefig,
    setup_style,
)
from driftfdr import RawThreshold, make_procedure, run_monitor, summarize
from driftfdr.datasets import elec2_scenario, insects_scenario

PROCEDURES = ["raw", "uncorrected", "bonferroni", "bh_window", "LORD++"]
ALPHA = 0.05
N_MODELS = 50


def make_scenarios(name, seed, quick):
    k = 12 if quick else N_MODELS
    if name == "insects":
        return {
            "extended": insects_scenario(n_models=k, seed=seed),
            "documented": insects_scenario(n_models=k, ground_truth="documented", seed=seed),
        }
    events = {"elec2_sporadic": 0, "elec2_events": 2}[name]
    return {"injected": elec2_scenario(n_models=k, drift_fraction=0.2, flip=0.5, drift_events=events, seed=seed)}


def run_task(args):
    data, detector_name, method, seed, quick = args
    scenarios = make_scenarios(data, seed, quick)
    main = next(iter(scenarios.values()))
    det = make_detector(detector_name)
    config = monitor_config(method if det.input_kind == "values" else "moving")
    cache, rows, timelines = {}, [], []
    for proc_name in PROCEDURES:
        proc = RawThreshold(det.default_threshold) if proc_name == "raw" else make_procedure(proc_name, ALPHA)
        result = run_monitor(main, det, proc, config, seed=seed, cache=cache)
        for truth, sc in scenarios.items():
            result.scenario = sc
            result.tests["is_null"] = sc.is_null(result.tests.stream, result.tests.ref_start, result.tests.t_end)
            rows.append({"data": data, "truth": truth, "detector": detector_name, "procedure": proc_name,
                         "seed": seed, **summarize(result)})
        if data == "insects" and seed == 0:
            per_window = result.tests.groupby("t_end").rejected.sum()
            timelines.append(pd.DataFrame({"procedure": proc_name, "detector": detector_name,
                                           "t_end": per_window.index, "alarms": per_window.to_numpy()}))
    return [{"kind": "summary", **r} for r in rows] + [
        {"kind": "timeline", **r} for t in timelines for r in t.to_dict("records")
    ]


def calibration_task(args):
    detector_name, method, seed, quick = args
    sc = insects_scenario(n_models=12 if quick else N_MODELS, seed=seed)
    det = make_detector(detector_name)
    tests = run_monitor(sc, det, None, monitor_config(method if det.input_kind == "values" else "moving")).tests
    first_change = sc.changes()[0][0].min()
    null = tests[tests.t_end <= first_change]
    p = null.pvalue.to_numpy()
    return [{"kind": "calibration", "detector": detector_name, "seed": seed, "n_tests": p.size,
             **{f"far@{a:g}": float(np.mean(p <= a)) for a in (0.05, 0.01, 1e-3)},
             "raw_far": float(np.mean(null.statistic > det.default_threshold))}]


def dispatch(task):
    kind, args = task
    return run_task(args) if kind == "run" else calibration_task(args)


def plot_timeline(tl: pd.DataFrame, changes_doc, changes_ext):
    import matplotlib.pyplot as plt

    procs = ["uncorrected", "bonferroni", "bh_window", "LORD++"]
    fig, axes = plt.subplots(len(procs), 1, figsize=(12, 7.5), sharex=True, sharey=True)
    for ax, proc in zip(axes, procs):
        d = tl[(tl.procedure == proc) & (tl.detector == "PH")]
        ax.bar(d.t_end - WINDOW / 2, d.alarms, width=WINDOW, color=PROCEDURE_COLORS[proc], linewidth=0)
        for c in changes_doc:
            ax.axvline(c, color=TEXT_2, linewidth=1)
        for c in sorted(set(changes_ext) - set(changes_doc)):
            ax.axvline(c, color=PALETTE[7], linewidth=1)
        ax.set_ylabel(PROCEDURE_LABELS[proc], rotation=0, ha="right", va="center")
        ax.grid(axis="x", visible=False)
    axes[-1].set_xlabel("шаг потока (после обучающего отрезка)")
    fig.suptitle("INSECTS, 50 моделей, Page-Hinkley: сколько моделей тревожит в каждом окне", y=0.995, fontsize=12, fontweight="bold")
    fig.text(0.5, 0.945, "серые линии — документированные дрейфы, красные — начала блоков одного класса",
             ha="center", fontsize=9, color=TEXT_2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig(fig, "exp7_insects_timeline.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--method", default="sieve")
    args = parser.parse_args()
    q = args.quick
    seeds = range(1) if q else range(3)
    tasks = [("run", (data, det, args.method, s, q)) for data in ("insects", "elec2_sporadic", "elec2_events")
             for det in ("PH", "DDM") for s in seeds]
    tasks += [("cal", (det, args.method, s, q)) for det in ("PH", "DDM") for s in seeds]
    print(f"{len(tasks)} tasks")
    rows = run_parallel(dispatch, tasks)
    summary = rows[rows.kind == "summary"].dropna(axis=1, how="all")
    timeline = rows[rows.kind == "timeline"].dropna(axis=1, how="all")
    calib = rows[rows.kind == "calibration"].dropna(axis=1, how="all")
    summary.to_csv(RESULTS / "exp7_runs.csv", index=False)
    calib.to_csv(RESULTS / "exp7_calibration.csv", index=False)
    cols = ["alarms", "false_alarms", "fdp", "window_fdp", "false_alarms_per_window", "mdr", "mean_delay", "degraded_per_drift"]
    agg = summary.groupby(["data", "truth", "detector", "procedure"])[cols].mean().reset_index()
    order = {p: i for i, p in enumerate(PROCEDURES)}
    agg = agg.sort_values(["data", "truth", "detector", "procedure"], key=lambda s: s.map(order) if s.name == "procedure" else s)
    agg.to_csv(RESULTS / "exp7_real.csv", index=False)
    cal = calib.groupby("detector")[["n_tests", "far@0.05", "far@0.01", "far@0.001", "raw_far"]].mean().reset_index()

    sc_ext = insects_scenario(n_models=2)
    sc_doc = insects_scenario(n_models=2, ground_truth="documented")
    setup_style()
    plot_timeline(timeline, sc_doc.changes()[0][0], sc_ext.changes()[0][0])
    with open(RESULTS / "exp7_tables.md", "w") as f:
        f.write(f"Метод калибровки для непрерывных сигналов: `{args.method}`.\n\n")
        f.write("## A. Калибровка на первом стабильном концепте INSECTS\n\n" + markdown_table(cal) + "\n\n")
        f.write("## B. Правила принятия решений, α = 0.05, 50 моделей\n\n" + markdown_table(agg) + "\n")
    print(cal.round(4).to_string(index=False))
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
