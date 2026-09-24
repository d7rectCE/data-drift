"""Experiment 19: runtime of driftfdr compared with river, Evidently and NannyML.

Everything runs in one process on one core, window = 100 steps, reference = 300
steps, horizon = 5 windows, on continuous N(0, 1) observations (the error of
the model; binary errors cost the same).

* Calibration: time to calibrate one model, i.e. what every (re)training costs,
  for each detector with the library defaults (B = 2000, sieve_pu, GPD tail)
  and with the lighter setting the experiments use (B = 500, exponential tail).
* Monitoring: time per step per model after calibration, amortised over window
  ends, against the uncalibrated ``river`` detectors.
* Scaling: ``StreamingMonitor`` with 10, 100 and 1000 models, with and without
  ``split_common``: total calibration time and time per step for the whole fleet
  (independent streams without drift; a step that ends in an alarm also pays
  for the recalibration, counted separately).
* Other tools (only if installed): one Evidently ``ValueDrift`` report per window
  and one NannyML ``PerformanceCalculator`` per model, as in experiment 17.

Usage: python experiments/exp19_runtime.py [--quick]
"""

from __future__ import annotations

import argparse
import platform
import time
import warnings

import numpy as np
import pandas as pd

from common import RESULTS, markdown_table
from driftfdr import (
    ADWIN,
    CalibratedDetector,
    CalibrationConfig,
    KSWindow,
    MeanShift,
    PageHinkley,
    StreamingMonitor,
)

N_REF, WINDOW, HORIZON = 300, 100, 5
DETECTORS = {
    "PH": PageHinkley,
    "KS": KSWindow,
    "ADWIN": ADWIN,
    "MeanShift(1)": lambda: MeanShift(1),
    "MeanShift(3)": lambda: MeanShift(3),
}
CONFIGS = {
    "B = 2000, GPD (по умолчанию)": CalibrationConfig(),
    "B = 500, экспонента": CalibrationConfig(n_boot=500, tail="exponential"),
}


def calibration_and_update(n_repeats: int, n_steps: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for det_name, factory in DETECTORS.items():
        for cfg_name, cfg in CONFIGS.items():
            cal_times, step_times = [], []
            for r in range(n_repeats):
                det = CalibratedDetector(factory(), N_REF, WINDOW, HORIZON, cfg, seed=r)
                ref, live = rng.normal(size=N_REF), rng.normal(size=n_steps)
                t0 = time.perf_counter()
                for v in ref:
                    det.update(v)
                cal_times.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                for v in live:
                    det.update(v)
                step_times.append((time.perf_counter() - t0) / n_steps)
            rows.append({"detector": det_name, "calibration": cfg_name,
                         "calibration_s": float(np.median(cal_times)),
                         "update_us_per_step": float(np.median(step_times)) * 1e6})
    return pd.DataFrame(rows)


def river_update(n_steps: int) -> pd.DataFrame:
    try:
        from river import drift
    except ImportError:
        return pd.DataFrame()
    x = np.random.default_rng(1).normal(size=n_steps)
    rows = []
    for name, det in (("river PageHinkley", drift.PageHinkley()), ("river ADWIN", drift.ADWIN())):
        t0 = time.perf_counter()
        for v in x:
            det.update(v)
        rows.append({"detector": name, "calibration": "нет", "calibration_s": 0.0,
                     "update_us_per_step": (time.perf_counter() - t0) / n_steps * 1e6})
    return pd.DataFrame(rows)


def fleet_scaling(sizes, n_steps: int) -> pd.DataFrame:
    rows = []
    warm = StreamingMonitor(5, PageHinkley, n_ref=N_REF, window=WINDOW, horizon=HORIZON)  # first-call overheads
    for v in np.random.default_rng(0).normal(size=(N_REF + 2 * WINDOW, 5)):
        warm.update(v)
    for k in sizes:
        rng = np.random.default_rng(k)
        x = rng.normal(size=(N_REF + n_steps, k))  # no drift, so step times rarely include a recalibration
        for split in (False, True):
            mon = StreamingMonitor(k, PageHinkley, "bonferroni", 0.05, n_ref=N_REF, window=WINDOW,
                                   horizon=HORIZON, split_common=split)
            t0 = time.perf_counter()
            for t in range(N_REF):
                mon.update(x[t])
            cal = time.perf_counter() - t0
            retrains = 0
            t0 = time.perf_counter()
            for t in range(N_REF, N_REF + n_steps):
                retrains += len(mon.update(x[t])) + mon.fleet_alarm
            step = (time.perf_counter() - t0) / n_steps
            rows.append({"models": k, "split_common": split, "calibration_total_s": cal, "recalibrations": retrains,
                         "ms_per_step_all_models": step * 1e3, "us_per_step_per_model": step / k * 1e6})
    return pd.DataFrame(rows)


def other_tools(n_windows: int) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    ref = rng.integers(0, 2, N_REF)
    windows = [rng.integers(0, 2, WINDOW) for _ in range(n_windows)]
    rows = []
    # driftfdr on the same 0/1 errors, amortised per window
    det = CalibratedDetector(MeanShift(1), N_REF, WINDOW, 1, CalibrationConfig(), seed=0)
    t0 = time.perf_counter()
    for v in ref:
        det.update(v)
    cal = time.perf_counter() - t0
    t0 = time.perf_counter()
    for w in windows:
        for v in w:
            det.update(v)
    rows.append({"tool": "driftfdr, MeanShift(1)", "setup_s": cal,
                 "ms_per_window": (time.perf_counter() - t0) / n_windows * 1e3})
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.metrics import ValueDrift

        dd = DataDefinition(numerical_columns=["err"])
        ref_ds = Dataset.from_pandas(pd.DataFrame({"err": ref}), data_definition=dd)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Report([ValueDrift(column="err")]).run(Dataset.from_pandas(pd.DataFrame({"err": windows[0]}), data_definition=dd), ref_ds)
            t0 = time.perf_counter()
            for w in windows:
                Report([ValueDrift(column="err")]).run(Dataset.from_pandas(pd.DataFrame({"err": w}), data_definition=dd), ref_ds)
        rows.append({"tool": "Evidently, ValueDrift", "setup_s": 0.0,
                     "ms_per_window": (time.perf_counter() - t0) / n_windows * 1e3})
    except ImportError:
        pass
    try:
        import nannyml as nml

        def frame(e):
            return pd.DataFrame({"y_pred": np.ones(e.size, int), "y_true": 1 - e, "y_pred_proba": np.full(e.size, 0.9)})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            calc = nml.PerformanceCalculator(y_pred="y_pred", y_true="y_true", y_pred_proba="y_pred_proba",
                                             metrics=["accuracy"], chunk_size=WINDOW, problem_type="classification_binary")
            calc.fit(frame(ref))
            setup = time.perf_counter() - t0
            t0 = time.perf_counter()
            calc.calculate(frame(np.concatenate(windows)))
            batch = time.perf_counter() - t0
            t0 = time.perf_counter()
            for w in windows[:20]:  # a live service calls it on each new window
                calc.calculate(frame(w))
            live = (time.perf_counter() - t0) / 20
        rows.append({"tool": "NannyML, все окна одним вызовом", "setup_s": setup, "ms_per_window": batch / n_windows * 1e3})
        rows.append({"tool": "NannyML, вызов на каждое окно", "setup_s": setup, "ms_per_window": live * 1e3})
    except ImportError:
        pass
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    repeats, steps, sizes, fleet_steps, windows = (
        (2, 1000, (10, 100), 300, 20) if args.quick else (10, 5000, (10, 100, 1000), 1000, 100))

    det = pd.concat([calibration_and_update(repeats, steps), river_update(20 * steps)], ignore_index=True)
    fleet = fleet_scaling(sizes, fleet_steps)
    tools = other_tools(windows)
    det.to_csv(RESULTS / "exp19_detectors.csv", index=False)
    fleet.to_csv(RESULTS / "exp19_fleet.csv", index=False)
    tools.to_csv(RESULTS / "exp19_tools.csv", index=False)

    machine = f"{platform.processor() or platform.machine()}, Python {platform.python_version()}, один процесс"
    with open(RESULTS / "exp19_tables.md", "w") as f:
        f.write(f"## Время работы ({machine})\n\nОкно {WINDOW} шагов, опорный отрезок {N_REF}, горизонт {HORIZON}; "
                f"медиана по {repeats} повторам.\n\n### Калибровка одной модели и шаг мониторинга\n\n"
                + markdown_table(det) + "\n\n### Парк моделей, Page-Hinkley, настройки по умолчанию\n\n"
                + markdown_table(fleet) + "\n\n### Сравнение с другими инструментами, ошибки 0/1, на одну модель\n\n"
                + markdown_table(tools) + "\n")
    for name, table in (("detectors", det), ("fleet", fleet), ("tools", tools)):
        print(f"\n{name}\n" + table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
