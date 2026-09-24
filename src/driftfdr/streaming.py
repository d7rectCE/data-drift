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

import json
import warnings
import zlib
from collections import deque
from dataclasses import asdict

import numpy as np

from .calibration import CalibrationConfig, NullDistribution, calibrate_many
from .detectors import ADWIN, DDM, Detector, PageHinkley
from .online_fdr import Procedure, make_procedure


class CalibratedDetector:
    """One detector for one model: collects a reference, calibrates, then emits p-values.

    ``update(x)`` returns ``None`` while the reference is being collected and between
    window ends, and a p-value at the end of every window afterwards. The statistic
    looks back over up to ``horizon`` windows; one bootstrap pass calibrates all of
    them. ``reset()`` starts a new reference, e.g. after retraining.
    """

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
        """Whether the reference is complete and the null distributions are ready."""
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

    The default rule is Bonferroni within each window, which keeps the share of false
    retrains near alpha however strongly the models' errors are correlated
    (experiment 13); ``"bh_window"`` reacts faster when many models drift at once
    (experiment 4) but loses control under strong correlation. Monitor the model's
    error, not its input features: feature detectors alarm on drift that does not
    hurt the model and miss drift that does (experiment 11).

    ``update`` takes the new observations, either a sequence with one value per model
    (models ``0..n-1``) or a mapping ``{model_id: value}`` for any subset of models,
    and returns the models that alarmed (indices or ids respectively). Each model keeps
    its own clock: its windows end after its own observations, so models that report
    irregularly or skip steps are fine, and p-values that become ready in the same
    call are corrected together. Alarmed models are reset and collect a new reference.
    With delayed labels, pass a model's error when its label arrives.

    Models can be added and removed at any time, and the whole state saved to and
    restored from a file (``save`` / ``load``).
    """

    def __init__(
        self,
        n_models: int | None = None,
        detector_factory=None,
        procedure: Procedure | str = "bonferroni",
        alpha: float = 0.05,
        n_ref: int = 300,
        window: int = 100,
        horizon: int = 5,
        calibration: CalibrationConfig = CalibrationConfig(),
        seed: int = 0,
        model_ids=None,
    ):
        if n_ref % window:
            raise ValueError("n_ref must be a multiple of window")
        if detector_factory is None:
            raise ValueError("detector_factory is required")
        self.detector_factory = detector_factory
        self.procedure_name = procedure if isinstance(procedure, str) else None
        self.procedure = make_procedure(procedure, alpha) if isinstance(procedure, str) else procedure
        self.alpha = alpha
        self.n_ref, self.window, self.horizon = n_ref, window, horizon
        self.calibration = calibration
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if isinstance(detector_factory(), ADWIN) and (n_models or 0) + len(model_ids or ()) > 1:
            warnings.warn(
                "ADWIN's calibrated p-values are about three times too small at the per-model "
                "levels a multiplicity correction uses (experiment 10); prefer MeanShift, "
                "PageHinkley or KSWindow when the false-alarm rate must hold.",
                stacklevel=2,
            )
        self.models: dict = {}
        self.n_seen: dict = {}
        ids = list(model_ids) if model_ids is not None else list(range(n_models or 0))
        for model_id in ids:
            self.add_model(model_id)

    def _seed(self, model_id, ref_start):
        key = model_id if isinstance(model_id, (int, np.integer)) else zlib.crc32(str(model_id).encode())
        return [self.calibration.seed, int(key), int(ref_start)]

    def add_model(self, model_id) -> None:
        """Start monitoring a new model; it first collects its reference."""
        if model_id in self.models:
            raise KeyError(f"model {model_id!r} is already monitored")
        self.models[model_id] = CalibratedDetector(
            self.detector_factory(), self.n_ref, self.window, self.horizon, self.calibration,
            seed=self._seed(model_id, 0),
        )
        self.n_seen[model_id] = 0

    def remove_model(self, model_id) -> None:
        """Stop monitoring a model and forget its state."""
        del self.models[model_id]
        del self.n_seen[model_id]

    def reset_model(self, model_id) -> None:
        """Call after retraining a model for any reason (alarmed models are reset automatically)."""
        self.models[model_id].reset(seed=self._seed(model_id, self.n_seen[model_id]))

    def update(self, observations):
        """Feed new observations; return the models to retrain now.

        ``observations`` is a sequence with one value per model (models ``0..n-1``,
        returns an integer array) or a mapping ``{model_id: value}`` for any subset
        of models (returns a list of ids). Alarmed models are reset automatically.
        """
        as_mapping = hasattr(observations, "items")
        items = observations.items() if as_mapping else enumerate(np.asarray(observations, dtype=float))
        ready, pvals = [], []
        for model_id, x in items:
            p = self.models[model_id].update(x)
            self.n_seen[model_id] += 1
            if p is not None:
                ready.append(model_id)
                pvals.append(p)
        alarmed = []
        if ready:
            rejected = self.procedure.decide(np.array(pvals), self.rng)
            alarmed = [m for m, r in zip(ready, rejected) if r]
            for model_id in alarmed:
                self.reset_model(model_id)
        return alarmed if as_mapping else np.array(alarmed, dtype=int)

    # --- persistence ---------------------------------------------------------

    def save(self, path) -> None:
        """Write the full state to an ``.npz`` file (arrays plus JSON metadata, no pickle)."""
        arrays, models = {}, []
        for i, (model_id, det) in enumerate(self.models.items()):
            entry = {
                "id": model_id,
                "n_seen": self.n_seen[model_id],
                "seed": det.seed,
                "since_reference": det._since_reference,
                "nulls": None,
            }
            arrays[f"m{i}_reference"] = np.asarray(det._reference, dtype=float)
            arrays[f"m{i}_recent"] = np.asarray(det._recent, dtype=float)
            if det._nulls is not None:
                entry["nulls"] = []
                for h, null in enumerate(det._nulls):
                    arrays[f"m{i}_null{h}"] = null.samples
                    entry["nulls"].append(
                        {k: _plain(getattr(null, k)) for k in (
                            "block_length", "min_exceedances", "tail_threshold", "tail_scale", "tail_shape", "tail_prob")}
                    )
            models.append(entry)
        meta = {
            "version": 1,
            "alpha": self.alpha,
            "n_ref": self.n_ref,
            "window": self.window,
            "horizon": self.horizon,
            "seed": self.seed,
            "calibration": _plain(asdict(self.calibration)),
            "procedure": {"name": self.procedure.name, "state": _plain(vars(self.procedure))},
            "rng": _plain(self.rng.bit_generator.state),
            "models": models,
        }
        np.savez(path, meta=np.array(json.dumps(meta)), **arrays)

    @classmethod
    def load(cls, path, detector_factory, procedure: Procedure | None = None) -> "StreamingMonitor":
        """Restore a monitor saved with ``save``. Custom procedures must be passed in again."""
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            arrays = {k: data[k] for k in data.files if k != "meta"}
        cal = CalibrationConfig(**meta["calibration"])
        proc_meta = meta["procedure"]
        mon = cls(
            detector_factory=detector_factory,
            procedure=procedure if procedure is not None else proc_meta["name"],
            alpha=meta["alpha"], n_ref=meta["n_ref"], window=meta["window"], horizon=meta["horizon"],
            calibration=cal, seed=meta["seed"], model_ids=[],
        )
        if procedure is None:
            for k, v in proc_meta["state"].items():
                setattr(mon.procedure, k, v)
        mon.rng.bit_generator.state = meta["rng"]
        for i, entry in enumerate(meta["models"]):
            model_id = entry["id"]
            mon.add_model(model_id)
            det = mon.models[model_id]
            det.seed = entry["seed"]
            det._reference = arrays[f"m{i}_reference"].tolist()
            det._recent.extend(arrays[f"m{i}_recent"].tolist())
            det._since_reference = entry["since_reference"]
            mon.n_seen[model_id] = entry["n_seen"]
            if entry["nulls"] is not None:
                det._nulls = [
                    NullDistribution(samples=arrays[f"m{i}_null{h}"], **params)
                    for h, params in enumerate(entry["nulls"])
                ]
        return mon


def _plain(obj):
    """Convert numpy scalars, arrays and tuples into JSON-serialisable Python values."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, deque)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


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
