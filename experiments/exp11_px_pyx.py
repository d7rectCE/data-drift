"""Experiment 11: which detectors see which drift, p(X) versus p(y|X).

Each stream carries a feature x, a label y and a fixed linear model. Virtual
drift shifts the distribution of x but leaves the model right; real drift
changes the slope and raises the error; "both" does both; cyclic drift
switches the slope on and off every 1000 steps. Detectors on the feature
(KS, two-sided Page-Hinkley) are compared with detectors on the model error
(Page-Hinkley and persistent MeanShift on the squared residual, DDM on 0/1
errors), all calibrated and corrected by Bonferroni within each window.

For each drift kind the table reports the share of streams that got at least
one alarm after the onset and the delay to it. Alarms on virtual drift are
needless retrains: the model did not get worse.

Usage: python experiments/exp11_px_pyx.py [--quick]
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from common import RESULTS, markdown_table, run_parallel
from driftfdr import (
    DDM,
    CalibrationConfig,
    KSWindow,
    MeanShift,
    MonitorConfig,
    PageHinkley,
    SupervisedConfig,
    make_procedure,
    make_supervised_scenario,
    run_monitor,
)

KINDS = ("virtual", "real", "both", "cyclic")
DETECTORS = {
    "KS на признаке": ("features", lambda: KSWindow()),
    "PH на признаке (двусторонний)": ("features", lambda: PageHinkley(mode="both")),
    "PH на потере": ("loss", lambda: PageHinkley()),
    "MeanShift(3) на потере": ("loss", lambda: MeanShift(3)),
    "DDM на ошибках 0/1": ("errors", lambda: DDM()),
}


def task(args):
    seed, n_streams = args
    sc = make_supervised_scenario(SupervisedConfig(n_streams=n_streams, drift_fraction=0.8, kinds=KINDS), seed=seed)
    rows = []
    for name, (signal, factory) in DETECTORS.items():
        view = replace(sc, values=sc.features) if signal == "features" else sc
        method = "moving" if signal == "errors" else "sieve"
        cfg = MonitorConfig(calibration=CalibrationConfig(method=method))
        t = run_monitor(view, factory(), make_procedure("bonferroni", 0.05), cfg, seed=seed).tests
        alarms = t[t.rejected]
        for kind in ("none",) + KINDS:
            streams = np.flatnonzero(sc.drift_kind == kind)
            onset = sc.change_start[streams] if kind != "none" else np.full(streams.size, 300)
            hit, delays = 0, []
            for k, tau in zip(streams, onset):
                after = alarms.t_end[(alarms.stream == k) & (alarms.t_end > tau)]
                if not after.empty:
                    hit += 1
                    delays.append(after.min() - tau)
            rows.append({"detector": name, "kind": kind, "seed": seed, "streams": streams.size,
                         "alarmed_share": hit / max(streams.size, 1),
                         "mean_delay": float(np.mean(delays)) if delays else np.nan})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n_streams, seeds = (50, 1) if args.quick else (300, 3)
    raw = run_parallel(task, [(s, n_streams) for s in range(seeds)])
    raw.to_csv(RESULTS / "exp11_runs.csv", index=False)
    agg = raw.groupby(["detector", "kind"])[["alarmed_share", "mean_delay"]].mean().reset_index()
    share = agg.pivot(index="detector", columns="kind", values="alarmed_share")[["none"] + list(KINDS)]
    delay = agg.pivot(index="detector", columns="kind", values="mean_delay")[list(KINDS)]
    share = share.loc[list(DETECTORS)].reset_index()
    delay = delay.loc[list(DETECTORS)].reset_index()
    share.to_csv(RESULTS / "exp11_share.csv", index=False)
    with open(RESULTS / "exp11_tables.md", "w") as f:
        f.write("## Доля моделей с тревогой после начала дрейфа (Бонферрони в окне, α = 0.05)\n\n")
        f.write("`none` — модели без дрейфа (ложные тревоги); `virtual` — дрейф p(X) без ухудшения модели "
                "(тревоги здесь — лишние переобучения).\n\n")
        f.write(markdown_table(share) + "\n\n## Средняя задержка до первой тревоги, шагов\n\n" + markdown_table(delay) + "\n")
    print(share.round(3).to_string(index=False))
    print(delay.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
