"""Experiment 23: detecting faster at the same false-alarm budget.

Three ways to cut the delay, all run through ``StreamingMonitor`` step by step
with Bonferroni and the same budget of false alarms per 100 steps (alpha = 0.05
per 100-step window across the fleet):

* AR prewhitening (``Prewhitened``): the detector runs on standardised AR
  innovations fitted on the reference;
* shorter windows: 25 steps with alpha / 4, so the budget per 100 steps is the same;
* a sequential e-detector (``ECUSUM``) checked at every step
  (``sequential=True``), against a windowed e-CUSUM and the windowed detectors.

Synthetic part: K = 100 AR(1) models (phi = 0.5), abrupt or gradual drift of 0.5 or
1 sd in 10% of the models, rho = 0 or 0.6, 2 seeds per case, plus drift-free
scenarios for the false-alarm rate. A drifting model's first alarm after its onset
is a detection, every other alarm is false. Real part (``--fx DIR``): the 40 FX
volatility models of experiment 22; an alarm is false if the model's mean loss over
the next 60 days exceeds the current reference by no more than delta, and
"degraded days" counts the days on which a model is materially worse than its
reference and has not been reset yet.

Usage: python experiments/exp23_speed.py [--quick] [--fx DATA_DIR]
"""

from __future__ import annotations

import argparse
import glob
import time

import numpy as np
import pandas as pd

from common import RESULTS, markdown_table, run_parallel
from driftfdr import (
    ECUSUM,
    CalibrationConfig,
    MeanShift,
    PageHinkley,
    Prewhitened,
    ScenarioConfig,
    StreamingMonitor,
    make_scenario,
)
from driftfdr import streaming
from driftfdr.datasets import forward_error, fx_scenario
from driftfdr.streams import NO_CHANGE

ALPHA, MAX_DELAY = 0.05, 1000
METHODS = {  # name: (factory, window, horizon, alpha scale, sequential)
    "PH, окно 100": (PageHinkley, 100, 5, 1.0, False),
    "MeanShift(3), окно 100": (lambda: MeanShift(3), 100, 5, 1.0, False),
    "PH + AR, окно 100": (lambda: Prewhitened(PageHinkley()), 100, 5, 1.0, False),
    "PH, окно 25": (PageHinkley, 25, 20, 0.25, False),
    "PH + AR, окно 25": (lambda: Prewhitened(PageHinkley()), 25, 20, 0.25, False),
    "e-CUSUM, окно 100": (ECUSUM, 100, 5, 1.0, False),
    "e-CUSUM, каждый шаг": (ECUSUM, 100, 5, 1.0, True),
}
CASES = {f"{kind}, Δ={mag:g}, ρ={rho:g}": dict(drift_type=kind, magnitude=mag, rho=rho, drift_fraction=0.1)
         for kind in ("abrupt", "gradual") for mag in (0.5, 1.0) for rho in (0.0, 0.6)}
CASES["no drift, ρ=0.3"] = dict(rho=0.3, drift_fraction=0.0)


_CALIBRATION_TIME = [0.0]
_original_calibrate = streaming.CalibratedDetector._calibrate


def _timed_calibrate(self):
    t0 = time.perf_counter()
    try:
        return _original_calibrate(self)
    finally:
        _CALIBRATION_TIME[0] += time.perf_counter() - t0


streaming.CalibratedDetector._calibrate = _timed_calibrate


def run(values, method, n_ref, calibration, seed, window=None, horizon=None):
    """Alarms ``[(model, step)]`` and the monitoring time per model-step in µs, calibration excluded."""
    factory, w, h, scale, sequential = METHODS[method]
    w, h = (window, horizon) if window else (w, h)
    mon = StreamingMonitor(values.shape[0], factory, "bonferroni", ALPHA * scale, n_ref=n_ref, window=w, horizon=h,
                           calibration=calibration, seed=seed, sequential=sequential)
    _CALIBRATION_TIME[0] = 0.0
    alarms, t0 = [], time.perf_counter()
    for t in range(values.shape[1]):
        alarms += [(int(k), t + 1) for k in mon.update(values[:, t])]
    total = time.perf_counter() - t0 - _CALIBRATION_TIME[0]
    return alarms, total / (values.shape[0] * values.shape[1]) * 1e6


def synthetic_task(args):
    case, seed, method, n = args
    sc = make_scenario(ScenarioConfig(n_streams=n, n_steps=5000, phi=0.5, gradual_length=500, **CASES[case]), seed=seed)
    alarms, us = run(sc.values, method, 300, CalibrationConfig(), seed)
    onset = {int(k): int(sc.change_start[k]) for k in sc.drifting}
    caught, false, delays, degraded, hits = set(), [], [], [], 0
    for k, t in alarms:
        if k in onset and t > onset[k] and k not in caught:
            caught.add(k)
            delays.append(t - onset[k])
            hits += t - onset[k] <= MAX_DELAY
        else:
            false.append(t)
    for k, o in onset.items():
        if k not in caught:
            degraded.append(sc.n_steps - o)
    degraded += delays
    windows = (sc.n_steps - 300) // 100
    precision = hits / len(alarms) if alarms else np.nan
    recall = hits / len(onset) if onset else np.nan
    return [{"case": case, "seed": seed, "method": method, "false_alarms": len(false),
             "p_false_alarm_per_100_steps": len({(t - 300) // 100 for t in false}) / windows,
             "mean_delay": float(np.mean(delays)) if delays else np.nan,
             "degraded_per_drift": float(np.mean(degraded)) if degraded else np.nan,
             "missed": 1 - len(caught) / len(onset) if onset else np.nan,
             "f1": (2 * precision * recall / (precision + recall) if hits else 0.0) if onset else np.nan,
             "us_per_model_step": us}]


FX_REF, FX_WINDOW, FX_HORIZON, FX_SPAN, FX_DELTA = 120, 20, 3, 60, 0.05


def fx_task(args):
    data_dir, method, seed = args
    sc = fx_scenario(sorted(glob.glob(f"{data_dir}/*.csv")))
    v = sc.values
    w = FX_WINDOW if METHODS[method][1] == 100 else FX_WINDOW // 4
    h = FX_HORIZON if METHODS[method][1] == 100 else FX_HORIZON * 4
    alarms, us = run(v, method, FX_REF, CalibrationConfig(tolerance=FX_DELTA), seed, window=w, horizon=h)
    fwd = forward_error(v, FX_SPAN)
    n, T = v.shape
    starts = {k: [0] for k in range(n)}
    false = 0
    for k, t in alarms:
        s = starts[k][-1]
        if fwd[k, t - 1] - v[k, s : s + FX_REF].mean() <= FX_DELTA:
            false += 1
        starts[k].append(t)
    degraded = 0
    for k in range(n):
        bounds = starts[k] + [T]
        for s, e in zip(bounds, bounds[1:]):
            ref = v[k, s : s + FX_REF].mean()
            lo = min(s + FX_REF, e)
            degraded += int(np.sum(fwd[k, lo:e] - ref > FX_DELTA))
    years = n * (T - FX_REF) / 252
    return [{"method": method, "seed": seed, "alarms_per_model_year": len(alarms) / years,
             "false_alarms_per_model_year": false / years, "fdp": false / len(alarms) if alarms else np.nan,
             "degraded_days_per_model_year": degraded / years, "us_per_model_step": us}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fx", default=None)
    args = parser.parse_args()
    n, seeds = (30, 1) if args.quick else (100, 2)
    tasks = [(c, s, m, n) for c in CASES for s in range(seeds) for m in METHODS]
    raw = run_parallel(synthetic_task, tasks)
    raw.to_csv(RESULTS / "exp23_runs.csv", index=False)
    cols = ["false_alarms", "p_false_alarm_per_100_steps", "mean_delay", "degraded_per_drift", "missed", "f1",
            "us_per_model_step"]
    drift = raw[~raw.case.str.startswith("no drift")]
    overall = drift.groupby("method", sort=False)[cols].mean().reset_index()
    null = raw[raw.case.str.startswith("no drift")].groupby("method", sort=False)[
        ["false_alarms", "p_false_alarm_per_100_steps"]].mean().reset_index()
    by_case = drift.pivot_table(index="case", columns="method", values="mean_delay", sort=False).reset_index()
    parts = [f"## Скорость обнаружения при одном бюджете ложных тревог (α = {ALPHA} на 100 шагов по парку, "
             f"Бонферрони), K = {n}\n\n### Сценарии с дрейфом, среднее\n\n" + markdown_table(overall),
             "### Без дрейфа: ложные тревоги\n\n" + markdown_table(null),
             "### Средняя задержка по случаям, шагов\n\n" + markdown_table(by_case)]
    if args.fx:
        fx = run_parallel(fx_task, [(args.fx, m, s) for m in METHODS for s in range(1 if args.quick else 3)])
        fx.to_csv(RESULTS / "exp23_fx_runs.csv", index=False)
        fx_agg = fx.groupby("method", sort=False).mean(numeric_only=True).drop(columns="seed").reset_index()
        parts.append(f"### Курсы валют, 40 моделей, δ = {FX_DELTA}, шаг — торговый день (окно 20 дней; "
                     "у методов с окном 25 — 5 дней)\n\n" + markdown_table(fx_agg))
        print(fx_agg.round(3).to_string(index=False))
    with open(RESULTS / "exp23_tables.md", "w") as f:
        f.write("\n\n".join(parts) + "\n")
    print(overall.round(3).to_string(index=False))
    print(null.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
