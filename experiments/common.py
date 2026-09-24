"""Shared setup for the experiment scripts: detectors, parallel runner, plotting style."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from driftfdr import (
    ADWIN,
    DDM,
    CalibrationConfig,
    KSWindow,
    MonitorConfig,
    PageHinkley,
    RawThreshold,
    make_procedure,
    make_scenario,
    run_monitor,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

N_REF, WINDOW, HORIZON = 300, 100, 5


def make_detector(name: str):
    return {
        "PH": PageHinkley,
        "DDM": DDM,
        "ADWIN": ADWIN,
        "KS": lambda: KSWindow(n_ref=N_REF, window=WINDOW),
    }[name]()


def default_resampling(detector) -> str:
    """AR-sieve for continuous signals, moving blocks for binary errors (see exp1)."""
    return "moving" if detector.input_kind == "errors" else "sieve"


def monitor_config(method: str, n_boot: int = 500, horizon: int = HORIZON) -> MonitorConfig:
    return MonitorConfig(
        n_ref=N_REF,
        window=WINDOW,
        horizon=horizon,
        calibration=CalibrationConfig(n_boot=n_boot, method=method),
    )


def run_procedures(scenario_config, seed, detector_name, procedures, n_boot=500):
    """All ``(name, alpha)`` procedures on one scenario, sharing calibrations."""
    scenario = make_scenario(scenario_config, seed)
    detector = make_detector(detector_name)
    config = monitor_config(default_resampling(detector), n_boot)
    cache = {}
    rows = []
    for name, alpha in procedures:
        if name == "raw":
            procedure = RawThreshold(detector.default_threshold)
        else:
            procedure = make_procedure(name, alpha)
        result = run_monitor(scenario, detector, procedure, config, seed=seed, cache=cache)
        rows.append({"procedure": name, "alpha": alpha, "seed": seed, **summarize(result)})
    return rows


def run_parallel(func, tasks, n_jobs=None):
    n_jobs = n_jobs or os.cpu_count()
    started = time.time()
    rows = []
    with Pool(n_jobs) as pool:
        for i, out in enumerate(pool.imap_unordered(func, tasks), 1):
            rows.extend(out)
            print(f"  {i}/{len(tasks)} tasks done, {time.time() - started:.0f}s", flush=True)
    return pd.DataFrame(rows)


# --- plotting ---------------------------------------------------------------

# Categorical slots in fixed order (validated palette); colour follows the entity.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
PROCEDURE_ORDER = ["LORD++", "uncorrected", "bh_window", "SAFFRON", "LOND", "bonferroni", "alpha-investing", "raw"]
PROCEDURE_COLORS = dict(zip(PROCEDURE_ORDER, PALETTE))
PROCEDURE_LABELS = {
    "LORD++": "LORD++",
    "uncorrected": "без поправки",
    "bh_window": "BH в окне",
    "SAFFRON": "SAFFRON",
    "LOND": "LOND",
    "bonferroni": "Бонферрони в окне",
    "alpha-investing": "alpha-investing",
    "raw": "детектор river по умолчанию",
    "storey_bh": "BH Стори в окне",
    "BatchBH": "BatchBH",
    "e_bh": "e-BH в окне",
}
# Stage 2 procedures reuse the slots of procedures that stage 2 figures do not show
PROCEDURE_COLORS["storey_bh"] = PALETTE[3]
PROCEDURE_COLORS["BatchBH"] = PALETTE[6]
PROCEDURE_COLORS["e_bh"] = PALETTE[4]
DETECTOR_ORDER = ["PH", "DDM", "ADWIN", "KS"]
DETECTOR_COLORS = dict(zip(DETECTOR_ORDER, PALETTE))

TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def setup_style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.handlelength": 1.6,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "lines.markersize": 5,
            "font.size": 9.5,
        }
    )


def line_kwargs(color):
    """2px line, >=8px markers with a surface-coloured ring."""
    return dict(
        color=color,
        linewidth=2,
        marker="o",
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
    )


def savefig(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def markdown_table(df: pd.DataFrame, floatfmt: str = "{:.3g}") -> str:
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "—" if np.isnan(v) else floatfmt.format(v)
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)
