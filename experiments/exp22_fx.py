"""Experiment 22: a fleet of volatility models on hourly FX data (2010-2026).

Five currency pairs (AUDUSD, EURUSD, GBPUSD, NZDUSD, USDCAD; MetaTrader 5 hourly
exports), eight models per pair forecasting the next-hour absolute return (see
``fx_scenario``): 40 models trained on 2010-2011 and monitored from 2012 by
their daily mean loss. Volatility regimes (the calm of 2012-2014, 2015, 2020,
2022) make the static models drift for real; nobody labelled the drifts, so an
alarm is judged by the forward oracle: the model's mean loss over the next 60
trading days exceeds its reference mean by more than delta.

Monitoring in trading days: reference 120 days, windows of 20 days, horizon 3.
Because the oracle looks 60 days ahead, a window is non-null before the loss
has risen, so the power per test understates detection; the main detection
metric is per episode (a run of consecutive non-null windows of one model):
the share of episodes with an alarm inside, and the median delay in days;
``long_*`` counts only persistent episodes of at least 3 windows (60 days).

Compared: river's Page-Hinkley at its default threshold; calibrated PH and
MeanShift(3) with no correction, Bonferroni and BH within each window; and
MeanShift(3) on the residuals of ``split_common`` with the common component
tested as one more stream (experiment 18).

The data are not part of the repository: pass the directory with the CSV files.

Usage: python experiments/exp22_fx.py DATA_DIR [--quick]
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import replace

import numpy as np
import pandas as pd

from common import RESULTS, markdown_table, run_parallel
from driftfdr import (
    CalibrationConfig,
    MeanShift,
    MonitorConfig,
    PageHinkley,
    RawThreshold,
    make_procedure,
    run_monitor,
    split_common,
    summarize,
)
from driftfdr.datasets import forward_error, fx_scenario
from driftfdr.streams import NO_CHANGE

N_REF, WINDOW, HORIZON, SPAN = 120, 20, 3, 60
DELTAS = (0.02, 0.05)
METHODS = [("PH", "raw", False), ("PH", "uncorrected", False), ("PH", "bonferroni", False), ("PH", "bh_window", False),
           ("MeanShift(3)", "uncorrected", False), ("MeanShift(3)", "bonferroni", False),
           ("MeanShift(3)", "bh_window", False), ("MeanShift(3)", "bonferroni", True), ("MeanShift(3)", "bh_window", True)]
_CACHE: dict = {}


def scenario(data_dir):
    if data_dir not in _CACHE:
        _CACHE[data_dir] = fx_scenario(sorted(glob.glob(f"{data_dir}/*.csv")))
    return _CACHE[data_dir]


def with_fleet(sc, delta):
    """Residual streams plus the common component as stream K, judged by the median model."""
    resid, common = split_common(sc.values, N_REF)
    n = sc.n_streams
    truth = np.vstack([sc.truth, np.median(sc.truth, axis=0)])
    truth_ref = np.vstack([sc.truth_ref, np.median(sc.truth_ref, axis=0)])
    return replace(sc, values=np.vstack([resid, common]), errors=np.vstack([sc.errors, common]),
                   change_start=np.append(sc.change_start, NO_CHANGE), change_end=np.append(sc.change_end, NO_CHANGE),
                   drift_kind=np.append(sc.drift_kind, "fleet"), event=np.append(sc.event, -1),
                   truth=truth, truth_ref=truth_ref, tolerance=delta)


SIZES = [0.05, 0.08, 0.12, 0.2, 1.0]


def episodes(tests, sc):
    """Runs of consecutive non-null tests per model, with their largest excess of the forward loss.

    Returns the share of episodes with an alarm inside, the median delay (days), the
    share among persistent episodes (at least 3 windows) and the share by excess size.
    """
    eps, delays = [], []
    for k, g in tests.sort_values("window").groupby("stream"):
        bad, rej, t_end = (~g.is_null).to_numpy(), g.rejected.to_numpy(), g.t_end.to_numpy()
        ref = np.array([sc.values[k, r : r + N_REF].mean() for r in g.ref_start])
        excess = sc.truth[k, t_end - WINDOW] - ref
        i = 0
        while i < len(bad):
            if not bad[i]:
                i += 1
                continue
            j = i
            while j < len(bad) and bad[j]:
                j += 1
            hits = np.flatnonzero(rej[i:j])
            eps.append((j - i, excess[i:j].max(), hits.size > 0))
            if hits.size:
                delays.append(t_end[i + hits[0]] - t_end[i])
            i = j
    e = pd.DataFrame(eps, columns=["length", "excess", "caught"])
    out = {"episodes": len(e), "episodes_caught": e.caught.mean(), "long_caught": e.caught[e.length >= 3].mean(),
           "episode_delay_days": float(np.median(delays)) if delays else np.nan}
    for lo, hi in zip(SIZES, SIZES[1:]):
        sel = (e.excess > lo) & (e.excess <= hi)
        out[f"caught, excess {lo:g}–{hi:g}"] = e.caught[sel].mean() if sel.any() else np.nan
    return out


def task(args):
    data_dir, delta, seed = args
    raw = scenario(data_dir)
    sc = raw.with_material_null(delta, truth=forward_error(raw.values, SPAN), truth_ref=raw.values)
    fleet_sc = with_fleet(sc, delta)
    n = sc.n_streams
    cfg = MonitorConfig(n_ref=N_REF, window=WINDOW, horizon=HORIZON, calibration=CalibrationConfig(tolerance=delta))
    rows = []
    for det_name, rule, split in METHODS:
        det = PageHinkley() if det_name == "PH" else MeanShift(3)
        proc = RawThreshold(det.default_threshold) if rule == "raw" else make_procedure(rule, 0.05)
        res = run_monitor(fleet_sc if split else sc, det, proc, cfg, seed=seed)
        t = res.tests
        fleet = t[(t.stream == n) & t.rejected] if split else t.iloc[:0]
        models = replace(res, tests=t[t.stream < n])
        s = summarize(models)
        ep = episodes(models.tests, sc)
        years = n * (sc.n_steps - N_REF) / 252
        rows.append({"delta": delta, "seed": seed, "detector": det_name,
                     "rule": ("river по умолчанию" if rule == "raw" else rule) + (" + split_common" if split else ""),
                     "alarms_per_model_year": s["alarms"] / years, "false_alarms_per_model_year": s["false_alarms"] / years,
                     "fdp": s["fdp"], **ep, "power_per_test": s["power_per_test"], "far_per_test": s["far_per_test"],
                     "p_any_false_alarm_per_window": s["p_any_false_alarm_per_window"],
                     "fleet_alarms": len(fleet), "fleet_false": int(fleet.is_null.sum()),
                     "non_null_share": float((~t[t.stream < n].is_null).mean())})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    seeds = 1 if args.quick else 3
    raw = run_parallel(task, [(args.data_dir, d, s) for d in DELTAS for s in range(seeds)])
    raw.to_csv(RESULTS / "exp22_runs.csv", index=False)
    cols = ["alarms_per_model_year", "false_alarms_per_model_year", "fdp", "episodes_caught", "long_caught",
            "episode_delay_days", "far_per_test", "p_any_false_alarm_per_window", "fleet_alarms", "fleet_false"]
    size_cols = [c for c in raw.columns if c.startswith("caught, excess")]
    agg = raw.groupby(["delta", "detector", "rule"], sort=False)[cols].mean().reset_index()
    sizes = raw[raw.delta == 0.05].groupby(["detector", "rule"], sort=False)[size_cols].mean().reset_index()
    agg.to_csv(RESULTS / "exp22_fx.csv", index=False)

    sc = scenario(args.data_dir)
    names = pd.Series(sc.drift_kind).str.split("/", expand=True)
    level = pd.DataFrame(sc.values.T, columns=sc.drift_kind)
    yearly = level.groupby(np.arange(sc.n_steps) // 252).mean().T
    yearly.index = pd.MultiIndex.from_frame(names, names=["pair", "model"])
    by_model = yearly.groupby(level="model", sort=False).mean().round(3)
    by_model.columns = [str(2012 + c) for c in by_model.columns]
    with open(RESULTS / "exp22_tables.md", "w") as f:
        f.write(f"## Парк из {sc.n_streams} моделей волатильности на часовых курсах, шаг — торговый день\n\n"
                f"Опора {N_REF} дней, окно {WINDOW}, горизонт {HORIZON}; истина — средняя потеря за следующие {SPAN} "
                "дней выше опорной больше чем на δ. Среднее по сидам бутстрепа.\n\n" + markdown_table(agg)
                + "\n\n### Доля пойманных эпизодов по размеру ухудшения (превышение над опорой), δ = 0.05\n\n"
                + markdown_table(sizes)
                + "\n\n### Средняя дневная потеря по годам (блоки по 252 торговых дня с января 2012), среднее по парам\n\n"
                + markdown_table(by_model.reset_index()) + "\n")
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
