"""Streaming interface: feed observations one step at a time, get p-values and alarms.

``CalibratedDetector`` wraps one detector for one model. It first collects a
reference segment, calibrates on it, and from then on returns a p-value at
the end of every check window (``None`` in between). Call ``reset()`` after
the model is retrained to collect a new reference.

``StreamingMonitor`` runs one calibrated detector per model and applies a
multiplicity rule to the p-values that arrive in the same window. It
reproduces ``run_monitor`` step by step, so the experiments describe exactly
what it does.

``from_river`` builds the equivalent vectorised detector from a configured
``river`` detector.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .calibration import CalibrationConfig, NullDistribution, calibrate_many
from .detectors import ADWIN, DDM, Detector, PageHinkley
from .online_fdr import Procedure, make_procedure


class CalibratedDetector:
    def __init__(
        self,
        detector: Detector,
        n_ref: int = 300,
        window: int = 100,
        horizon: int = 5,
        calibration: CalibrationConfig = CalibrationConfig(),
        seed=None,
    ):
        self.detector = detector
        self.n_ref, self.window, self.horizon = n_ref, window, horizon
        self.calibration = calibration
        self.seed = seed
        self.reset()

    def reset(self, seed=None):
        """Start collecting a new reference, e.g. right after retraining."""
        if seed is not None:
            self.seed = seed
        self._reference: list[float] = []
        self._recent: deque = deque(maxlen=self.horizon * self.window)
        self._nulls: list[NullDistribution] | None = None
        self._since_reference = 0

    @property
    def calibrated(self) -> bool:
        return self._nulls is not None

    def update(self, x: float) -> float | None:
        """Add one observation; returns a p-value at the end of each window after calibration."""
        if self._nulls is None:
            self._reference.append(float(x))
            if len(self._reference) == self.n_ref:
                self._nulls = self._calibrate()
            return None
        self._recent.append(float(x))
        self._since_reference += 1
        if self._since_reference % self.window:
            return None
        seen = min(self._since_reference // self.window, self.horizon)
        series = np.concatenate([self._reference, list(self._recent)[-seen * self.window :]])
        stat = self.detector.window_statistics(series[None], self.n_ref, self.window)[0, -1]
        return float(self._nulls[seen - 1].pvalue(stat)[0])

    def _calibrate(self):
        seed = self.seed if self.seed is not None else np.random.SeedSequence().entropy
        ref = np.asarray(self._reference)[None]
        return calibrate_many(self.detector, ref, self.window, self.horizon, self.calibration, [seed])[0]


class StreamingMonitor:
    """One calibrated detector per model plus a rule that decides which models to retrain.

    ``update`` takes the current observation of every model and returns the
    indices of models that alarmed at this step (empty between window ends).
    Alarmed models are reset automatically.
    """

    def __init__(
        self,
        n_models: int,
        detector_factory,
        procedure: Procedure | str = "bh_window",
        alpha: float = 0.05,
        n_ref: int = 300,
        window: int = 100,
        horizon: int = 5,
        calibration: CalibrationConfig = CalibrationConfig(),
        seed: int = 0,
    ):
        if n_ref % window:
            raise ValueError("n_ref must be a multiple of window")
        self.procedure = make_procedure(procedure, alpha) if isinstance(procedure, str) else procedure
        self.rng = np.random.default_rng(seed)
        self.calibration = calibration
        self.t = 0
        self.models = [
            CalibratedDetector(detector_factory(), n_ref, window, horizon, calibration, seed=self._seed(k, 0))
            for k in range(n_models)
        ]

    def _seed(self, k, ref_start):
        return [self.calibration.seed, k, ref_start]

    def update(self, xs) -> np.ndarray:
        xs = np.asarray(xs, dtype=float)
        pvals = {k: m.update(x) for k, (m, x) in enumerate(zip(self.models, xs))}
        self.t += 1
        ready = np.array([k for k, p in pvals.items() if p is not None], dtype=int)
        if ready.size == 0:
            return ready
        rejected = self.procedure.decide(np.array([pvals[k] for k in ready]), self.rng)
        alarmed = ready[rejected]
        for k in alarmed:
            self.models[k].reset(seed=self._seed(int(k), self.t))
        return alarmed


def from_river(river_detector) -> Detector:
    """Vectorised equivalent of a configured ``river.drift`` detector (PageHinkley, DDM, ADWIN)."""
    name = type(river_detector).__name__
    if name == "PageHinkley":
        return PageHinkley(
            delta=river_detector.delta,
            alpha=river_detector.alpha,
            min_instances=river_detector.min_instances,
            mode=river_detector.mode,
            threshold=river_detector.threshold,
        )
    if name == "DDM":
        return DDM(warm_start=river_detector.warm_start, drift_threshold=river_detector.drift_threshold)
    if name == "ADWIN":
        return ADWIN(
            delta=river_detector.delta,
            min_window_length=river_detector.min_window_length,
            grace_period=river_detector.grace_period,
        )
    raise ValueError(f"no vectorised equivalent for river detector {name}")
