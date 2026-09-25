"""Experiment 24: monitoring without labels through NannyML's CBPE.

When labels arrive late, the model's error is unknown; NannyML's CBPE estimates
it from the predicted probabilities. This checks whether driftfdr can monitor
that estimate instead of the true error, and what it misses.

Electricity, a fleet of 50 logistic models (as in experiment 7), in two
variants: natural changes only, and with real concept drift injected into 20%
of the models (their labels flipped with probability 0.3 after a random time,
as in ``elec2_scenario``). A step is a chunk of 100 rows; the reference is the
40 chunks after training (with labels, to fit CBPE); windows of 10 chunks,
horizon 4, tolerance delta = 5 points of error (the error of a chunk is noisy:
sd about 0.19, lag-1 autocorrelation about 0.6). Two signals are monitored with
the same detectors: the true error per chunk (labels available at once) and
1 - CBPE's estimated accuracy (no labels). An alarm is judged by the true error:
it is false if the true error over the next 20 chunks exceeds the reference by no
more than delta. Needs nannyml, river and the Elec2 data.

Usage: python experiments/exp24_cbpe.py [--quick]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import RESULTS, markdown_table, run_parallel
from driftfdr import ECUSUM, CalibrationConfig, MeanShift, MonitorConfig, make_procedure, run_monitor, summarize
from driftfdr.datasets import _load_river, _predict_proba, _scenario, fit_softmax, forward_error
from driftfdr.integrations import cbpe_estimated_error
from driftfdr.streams import NO_CHANGE, ScenarioConfig

TRAIN, CHUNK, N_REF, WINDOW, HORIZON, SPAN, DELTA = 5000, 100, 40, 10, 4, 20, 0.05
DETECTORS = {"MeanShift(1)": lambda: MeanShift(1), "e-CUSUM": ECUSUM}


def fleet(n_models, seed, inject):
    """True and CBPE-estimated error per chunk for every model, and the injected change chunks."""
    from river import datasets

    X, y = _load_river(datasets.Elec2(), drop=("date",))
    X = X[:, ~np.isnan(X).any(axis=0)]
    y = y.astype(int)
    rng = np.random.default_rng(seed)
    mu, sd = X[:TRAIN].mean(axis=0), X[:TRAIN].std(axis=0) + 1e-9
    Z = (X - mu) / sd
    n_chunks = (len(y) - TRAIN) // CHUNK
    end = TRAIN + n_chunks * CHUNK
    true, est, change = [], [], np.full(n_models, NO_CHANGE)
    drifting = rng.choice(n_models, size=int(0.2 * n_models), replace=False) if inject else []
    for k in range(n_models):
        cols = rng.choice(X.shape[1], size=min(4, X.shape[1]), replace=False)
        W = fit_softmax(Z[:TRAIN, cols], y[:TRAIN], 2)
        proba = _predict_proba(W, Z[:, cols])[:, 1]
        target = y.copy()
        if k in drifting:
            onset = int(rng.integers(N_REF + 20, n_chunks - 40))
            flip = rng.random(len(y) - TRAIN - onset * CHUNK) < 0.3
            target[TRAIN + onset * CHUNK :] = np.where(flip, 1 - y[TRAIN + onset * CHUNK :], y[TRAIN + onset * CHUNK :])
            change[k] = onset
        frame = pd.DataFrame({"y_pred_proba": proba, "y_pred": (proba > 0.5).astype(int), "y_true": target})
        after = frame.iloc[TRAIN:end]
        true.append((after.y_pred != after.y_true).to_numpy().reshape(n_chunks, CHUNK).mean(axis=1))
        est.append(cbpe_estimated_error(after.iloc[: N_REF * CHUNK], after.iloc[N_REF * CHUNK :], CHUNK))
    true = np.array(true)
    est = np.concatenate([true[:, :N_REF], np.array(est)], axis=1)  # the reference itself has labels
    return true, est, change


def episodes(tests):
    """Share of runs of consecutive non-null tests with an alarm inside (as in experiment 22)."""
    caught = total = 0
    for _, g in tests.sort_values("window").groupby("stream"):
        bad, rej = (~g.is_null).to_numpy(), g.rejected.to_numpy()
        i = 0
        while i < len(bad):
            if not bad[i]:
                i += 1
                continue
            j = i
            while j < len(bad) and bad[j]:
                j += 1
            total += 1
            caught += bool(rej[i:j].any())
            i = j
    return caught / total if total else np.nan


def task(args):
    variant, seed, n = args
    true, est, change = fleet(n, seed, variant != "естественные изменения")
    config = ScenarioConfig(n_streams=n, n_steps=true.shape[1], phi=np.nan, rho=np.nan, drift_fraction=np.nan)
    rows = []
    for signal, values in (("настоящая ошибка (с метками)", true), ("оценка CBPE (без меток)", est)):
        sc = _scenario(values, (true > 0.5).astype(np.int8), change[:, None], config)
        sc = sc.with_material_null(DELTA, truth=forward_error(true, SPAN), truth_ref=true)
        for det_name, factory in DETECTORS.items():
            cfg = MonitorConfig(n_ref=N_REF, window=WINDOW, horizon=HORIZON, calibration=CalibrationConfig(tolerance=DELTA))
            res = run_monitor(sc, factory(), make_procedure("bonferroni", 0.05), cfg, seed=seed)
            s = summarize(res)
            injected = res.tests[res.tests.stream.isin(np.flatnonzero(change < NO_CHANGE))]
            rows.append({"variant": variant, "seed": seed, "signal": signal, "detector": det_name,
                         "alarms": s["alarms"], "fdp": s["fdp"], "episodes_caught": episodes(res.tests),
                         "injected_drift_caught": episodes(injected) if len(injected) else np.nan,
                         "corr_with_true_error": float(np.mean([np.corrcoef(true[k, N_REF:], est[k, N_REF:])[0, 1]
                                                                for k in range(n)]))})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n, seeds = (10, 1) if args.quick else (50, 2)
    variants = ["естественные изменения", "внедрён дрейф p(y|X) у 20% моделей"]
    raw = run_parallel(task, [(v, s, n) for v in variants for s in range(seeds)])
    raw.to_csv(RESULTS / "exp24_runs.csv", index=False)
    cols = ["alarms", "fdp", "episodes_caught", "injected_drift_caught", "corr_with_true_error"]
    agg = raw.groupby(["variant", "signal", "detector"], sort=False)[cols].mean().reset_index()
    agg.to_csv(RESULTS / "exp24_cbpe.csv", index=False)
    with open(RESULTS / "exp24_tables.md", "w") as f:
        f.write(f"## Мониторинг без меток через CBPE, Electricity, {n} моделей, шаг — {CHUNK} строк, "
                f"Бонферрони, δ = {DELTA}\n\nИстина — настоящая ошибка за следующие {SPAN} шагов. "
                "`episodes_caught` — доля серий ненулевых окон с тревогой внутри; `injected_drift_caught` — то же "
                "только для моделей с внедрённым дрейфом.\n\n" + markdown_table(agg) + "\n")
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
