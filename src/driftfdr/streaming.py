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

    With ``sequential=True`` (detectors with ``start_stream``, such as ``ECUSUM``) the
    detector runs continuously from the end of the reference and ``update`` returns a
    p-value at every step: the p-value of the current score under the null of the
    largest score within a window. Alarming as soon as it drops below a level ``a``
    therefore keeps the probability of a false alarm within any window at ``a``, as in
    the windowed mode, without waiting for the window to end.
    """

    def __init__(
        self,
        detector: Detector,
        n_ref: int = 300,
        window: int = 100,
        horizon: int = 5,
        calibration: CalibrationConfig = CalibrationConfig(),
        seed=None,
        sequential: bool = False,
    ):
        if sequential and not hasattr(detector, "start_stream"):
            raise ValueError(f"{detector.name} cannot run sequentially; use a detector such as ECUSUM")
        self.detector = detector
        self.n_ref, self.window, self.horizon = n_ref, window, horizon
        self.calibration = calibration
        self.seed = seed
        self.sequential = sequential
        self.reset()

    def reset(self, seed=None):
        """Start collecting a new reference, e.g. right after retraining."""
        if seed is not None:
            self.seed = seed
        self._reference: list[float] = []
        self._recent: deque = deque(maxlen=self.horizon * self.window)
        self._nulls: list[NullDistribution] | None = None
        self._stream = None
        self._since_reference = 0

    @property
    def calibrated(self) -> bool:
        """Whether the reference is complete and the null distributions are ready."""
        return self._nulls is not None

    def update(self, x: float) -> float | None:
        """Add one observation; returns a p-value at the end of each window after calibration
        (at every step in sequential mode)."""
        out = self._push(x)
        if out is None or out[0] == "p":
            return None if out is None else out[1]
        seen = out[1]
        stat = self.detector.window_statistics(self._window_series(seen)[None], self.n_ref, self.window)[0, -1]
        return self._pvalue(stat, seen)

    def _push(self, x: float):
        """Add one observation. Returns ``None``, ``("p", pvalue)`` or ``("window", seen)``
        when a window statistic is due (so that a monitor can compute many at once)."""
        if self._nulls is None:
            self._reference.append(float(x))
            if len(self._reference) == self.n_ref:
                self._nulls = self._calibrate()
                if self.sequential:
                    self._stream = self.detector.start_stream(np.asarray(self._reference))
            return None
        self._since_reference += 1
        if self.sequential:
            if self._since_reference > 1 and (self._since_reference - 1) % self.window == 0:
                self._stream.roll(self.horizon)  # look back over at most `horizon` windows, as calibrated
            score = self._stream.update(float(x))
            seen = min(-(-self._since_reference // self.window), self.horizon)
            return "p", self._nulls[seen - 1].pvalue_scalar(score)
        self._recent.append(float(x))
        if self._since_reference % self.window:
            return None
        return "window", min(self._since_reference // self.window, self.horizon)

    def _window_series(self, seen: int) -> np.ndarray:
        return np.concatenate([self._reference, list(self._recent)[-seen * self.window :]])

    def _pvalue(self, stat: float, seen: int) -> float:
        return float(self._nulls[seen - 1].pvalue(stat)[0])

    def _calibrate(self):
        seed = self.seed if self.seed is not None else np.random.SeedSequence().entropy
        ref = np.asarray(self._reference)[None]
        return calibrate_many(self.detector, ref, self.window, self.horizon, self.calibration, [seed])[0]


def _config_key(obj):
    """Hashable description of a detector's class and settings, for batching equal detectors."""
    if isinstance(obj, Detector):
        return (type(obj).__name__, tuple(sorted((k, _config_key(v)) for k, v in vars(obj).items())))
    if isinstance(obj, (list, tuple)):
        return tuple(_config_key(v) for v in obj)
    return obj


def _finish(pushed):
    """Turn ``[(key, CalibratedDetector, _push result)]`` into ``[(key, pvalue)]``.

    Window statistics that are due are computed in one vectorised call per group of
    equal detectors with the same look-back, instead of one call per model.
    """
    out, groups = [], {}
    for i, (key, det, res) in enumerate(pushed):
        if res is None:
            continue
        if res[0] == "p":
            out.append((i, key, res[1]))
        else:
            groups.setdefault((res[1], _config_key(det.detector)), []).append((i, key, det))
    for (seen, _), members in groups.items():
        series = np.stack([det._window_series(seen) for _, _, det in members])
        det0 = members[0][2]
        stats = det0.detector.window_statistics(series, det0.n_ref, det0.window)[:, -1]
        out += [(i, key, det._pvalue(st, seen)) for (i, key, det), st in zip(members, stats)]
    out.sort(key=lambda t: t[0])
    return [(key, p) for _, key, p in out]


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

    ``split_common=True`` is for fleets whose errors move together (shared data source,
    shared features): a common fluctuation otherwise makes many models alarm at once,
    and with ``"bh_window"`` these bursts dominate the false alarms (experiment 18). Each
    model's values are standardised by the mean and standard deviation of its own
    reference; the cross-sectional median of the standardised values is the common
    component, and the detectors watch the residuals (value minus the median). The
    common component gets its own calibrated detector, tested in the same family as the
    models; when it alarms, ``fleet_alarm`` is set for that step: something shared by
    all models has changed, which is better handled as an incident than by retraining
    every model. A drift of the whole fleet is invisible in the residuals and is caught
    only this way; the median assumes fewer than half of the models drift at once. In
    this mode every monitored model must report at every step, and a retrained model
    re-estimates its standardisation from its new reference.

    ``sequential=True`` (with a detector such as ``ECUSUM``) removes the wait for the end
    of a window: every model is checked at every step, at level ``alpha / K`` against
    the null of the largest score within a window, so the probability of a false alarm
    in any window stays at most ``alpha`` across the fleet (a Bonferroni rule; the
    ``procedure`` argument is not used). ``window`` then only sets the time unit of
    that error budget.

    After every ``update``, ``last_pvalues`` holds the p-values that became ready in that
    call, keyed by model id (and ``StreamingMonitor.FLEET`` for the common component).
    """

    FLEET = "__fleet__"

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
        split_common: bool = False,
        sequential: bool = False,
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
        self.split_common = split_common
        self.sequential = sequential
        self.fleet_alarm = False
        self.last_pvalues: dict = {}
        self.fleet: CalibratedDetector | None = None
        self._step = 0  # synchronous steps seen (split_common only)
        self._scale: dict = {}  # model_id -> (mean, sd) of its reference
        self._pending: dict = {}  # model_id -> [(value, step)] while its reference is collected
        self._common: dict = {}  # step -> common component (nan if no model was standardised)
        if split_common:
            self.fleet = self._new_detector(self.FLEET, 0)
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
        self.models[model_id] = self._new_detector(model_id, 0)
        self.n_seen[model_id] = 0
        if self.split_common:
            self._pending[model_id] = []

    def _new_detector(self, model_id, ref_start) -> CalibratedDetector:
        return CalibratedDetector(
            self.detector_factory(), self.n_ref, self.window, self.horizon, self.calibration,
            seed=self._seed(model_id, ref_start), sequential=self.sequential,
        )

    def remove_model(self, model_id) -> None:
        """Stop monitoring a model and forget its state."""
        del self.models[model_id]
        del self.n_seen[model_id]
        self._scale.pop(model_id, None)
        self._pending.pop(model_id, None)

    def reset_model(self, model_id) -> None:
        """Call after retraining a model for any reason (alarmed models are reset automatically)."""
        self.models[model_id].reset(seed=self._seed(model_id, self.n_seen[model_id]))
        if self.split_common:
            self._scale.pop(model_id, None)
            self._pending[model_id] = []

    def update(self, observations):
        """Feed new observations; return the models to retrain now.

        ``observations`` is a sequence with one value per model (models ``0..n-1``,
        returns an integer array) or a mapping ``{model_id: value}`` for any subset
        of models (returns a list of ids). Alarmed models are reset automatically.
        """
        as_mapping = hasattr(observations, "items")
        items = observations.items() if as_mapping else enumerate(np.asarray(observations, dtype=float))
        if self.split_common:
            ready, pvals = self._split_step(dict(items))
        else:
            pushed = []
            for model_id, x in items:
                det = self.models[model_id]
                pushed.append((model_id, det, det._push(x)))
                self.n_seen[model_id] += 1
            done = _finish(pushed)
            ready, pvals = [k for k, _ in done], [p for _, p in done]
        self.last_pvalues = dict(zip(ready, pvals))
        self.fleet_alarm = False
        alarmed = []
        if ready:
            if self.sequential:  # every model at alpha / K at every step, see the class docstring
                n_tests = len(self.models) + (self.fleet is not None)
                rejected = np.array(pvals) <= self.alpha / n_tests
            else:
                rejected = self.procedure.decide(np.array(pvals), self.rng)
            alarmed = [m for m, r in zip(ready, rejected) if r]
            if self.FLEET in alarmed and self.split_common:
                alarmed.remove(self.FLEET)
                self.fleet_alarm = True
                self.fleet.reset(seed=self._seed(self.FLEET, self._step))
            for model_id in alarmed:
                self.reset_model(model_id)
        return alarmed if as_mapping else np.array(alarmed, dtype=int)

    def _split_step(self, obs: dict):
        """One synchronous step with the common component removed; returns (ids, p-values)."""
        missing = set(self.models) - set(obs)
        if missing:
            raise ValueError(f"with split_common every model must report at every step; missing {sorted(map(str, missing))}")
        for model_id in obs:
            if model_id not in self.models:
                raise KeyError(f"model {model_id!r} is not monitored")
        t = self._step
        self._step += 1
        z = {m: (float(obs[m]) - mu) / sd for m, (mu, sd) in self._scale.items()}
        common = float(np.median(list(z.values()))) if z else np.nan
        self._common[t] = common
        if np.isnan(common) and (self.fleet.calibrated or self.fleet._reference):
            # the common component has a gap: its detector starts over
            self.fleet.reset(seed=self._seed(self.FLEET, t))
        feed = {m: [zm - common] for m, zm in z.items()}

        completed = []
        for m in obs:
            self.n_seen[m] += 1
            if m not in self._scale:
                self._pending[m].append((float(obs[m]), t))
                if len(self._pending[m]) == self.n_ref:
                    completed.append(m)
        fleet_steps = [t]
        if completed:
            residuals, gaps = self._standardise(completed)
            feed.update(residuals)
            fleet_steps = sorted(set(gaps) | {t})
        for step in [s for s in self._common if s <= t - self.n_ref]:
            del self._common[step]

        pushed = []
        for m, residuals in feed.items():
            det, res = self.models[m], None
            for r in residuals:
                res = det._push(r)
            pushed.append((m, det, res))
        res = None
        for step in fleet_steps:
            if not np.isnan(self._common[step]):
                res = self.fleet._push(self._common[step]) or res
        pushed.append((self.FLEET, self.fleet, res))
        done = _finish(pushed)
        return [k for k, _ in done], [p for _, p in done]

    def _standardise(self, completed):
        """Models whose reference just completed: fix their scale and fill gaps in the common component.

        Returns the residuals of their reference steps and the steps whose common component was filled in.
        """
        raw = {m: np.array([x for x, _ in self._pending[m]]) for m in completed}
        steps = {m: [s for _, s in self._pending[m]] for m in completed}
        pos = {m: {s: i for i, s in enumerate(steps[m])} for m in completed}
        for m in completed:
            mu, sd = raw[m].mean(), raw[m].std()
            self._scale[m] = (float(mu), float(sd) if sd > 0 else 1.0)
        z = {m: (raw[m] - self._scale[m][0]) / self._scale[m][1] for m in completed}
        gaps = sorted({s for m in completed for s in steps[m] if np.isnan(self._common[s])})
        for s in gaps:  # no model was standardised then: the median over the models completing now
            self._common[s] = float(np.median([z[m][pos[m][s]] for m in completed if s in pos[m]]))
        out = {}
        for m in completed:
            out[m] = list(z[m] - np.array([self._common[s] for s in steps[m]]))
            self._pending[m] = []
        return out, gaps

    # --- persistence ---------------------------------------------------------

    def save(self, path) -> None:
        """Write the full state to an ``.npz`` file (arrays plus JSON metadata, no pickle)."""
        arrays, models = {}, []
        for i, (model_id, det) in enumerate(self.models.items()):
            entry = {"id": model_id, "n_seen": self.n_seen[model_id], **_save_detector(det, f"m{i}", arrays)}
            if self.split_common:
                entry["scale"] = self._scale.get(model_id)
                pending = self._pending[model_id]
                arrays[f"m{i}_pending"] = np.array([x for x, _ in pending], dtype=float)
                arrays[f"m{i}_pending_steps"] = np.array([s for _, s in pending], dtype=np.int64)
            models.append(entry)
        meta = {
            "version": 2,
            "alpha": self.alpha,
            "n_ref": self.n_ref,
            "window": self.window,
            "horizon": self.horizon,
            "seed": self.seed,
            "calibration": _plain(asdict(self.calibration)),
            "procedure": {"name": self.procedure.name, "state": _plain(vars(self.procedure))},
            "rng": _plain(self.rng.bit_generator.state),
            "models": models,
            "split_common": self.split_common,
            "sequential": self.sequential,
        }
        if self.split_common:
            meta["step"] = self._step
            meta["fleet"] = _save_detector(self.fleet, "fleet", arrays)
            steps = sorted(self._common)
            arrays["common_steps"] = np.array(steps, dtype=np.int64)
            arrays["common"] = np.array([self._common[s] for s in steps], dtype=float)
        np.savez(path, meta=np.array(json.dumps(meta)), **arrays)

    @classmethod
    def load(cls, path, detector_factory, procedure: Procedure | None = None) -> "StreamingMonitor":
        """Restore a monitor saved with ``save``. Custom procedures must be passed in again."""
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            arrays = {k: data[k] for k in data.files if k != "meta"}
        cal = CalibrationConfig(**meta["calibration"])
        proc_meta = meta["procedure"]
        split = meta.get("split_common", False)
        mon = cls(
            detector_factory=detector_factory,
            procedure=procedure if procedure is not None else proc_meta["name"],
            alpha=meta["alpha"], n_ref=meta["n_ref"], window=meta["window"], horizon=meta["horizon"],
            calibration=cal, seed=meta["seed"], model_ids=[], split_common=split,
            sequential=meta.get("sequential", False),
        )
        if procedure is None:
            for k, v in proc_meta["state"].items():
                setattr(mon.procedure, k, v)
        mon.rng.bit_generator.state = meta["rng"]
        for i, entry in enumerate(meta["models"]):
            model_id = entry["id"]
            mon.add_model(model_id)
            _load_detector(mon.models[model_id], entry, f"m{i}", arrays)
            mon.n_seen[model_id] = entry["n_seen"]
            if split:
                if entry["scale"] is not None:
                    mon._scale[model_id] = tuple(entry["scale"])
                mon._pending[model_id] = list(zip(arrays[f"m{i}_pending"].tolist(),
                                                  arrays[f"m{i}_pending_steps"].tolist()))
        if split:
            mon._step = meta["step"]
            _load_detector(mon.fleet, meta["fleet"], "fleet", arrays)
            mon._common = dict(zip(arrays["common_steps"].tolist(), arrays["common"].tolist()))
        return mon


def _save_detector(det: CalibratedDetector, prefix: str, arrays: dict) -> dict:
    """Arrays of one calibrated detector go into ``arrays``; returns its JSON metadata."""
    entry = {"seed": det.seed, "since_reference": det._since_reference, "nulls": None,
             "stream": det._stream.to_dict() if det._stream is not None else None}
    arrays[f"{prefix}_reference"] = np.asarray(det._reference, dtype=float)
    arrays[f"{prefix}_recent"] = np.asarray(det._recent, dtype=float)
    if det._nulls is not None:
        entry["nulls"] = []
        for h, null in enumerate(det._nulls):
            arrays[f"{prefix}_null{h}"] = null.samples
            entry["nulls"].append(
                {k: _plain(getattr(null, k)) for k in (
                    "block_length", "min_exceedances", "tail_threshold", "tail_scale", "tail_shape", "tail_prob")}
            )
    return _plain(entry)


def _load_detector(det: CalibratedDetector, entry: dict, prefix: str, arrays: dict) -> None:
    det.seed = entry["seed"]
    det._reference = arrays[f"{prefix}_reference"].tolist()
    det._recent.extend(arrays[f"{prefix}_recent"].tolist())
    det._since_reference = entry["since_reference"]
    if entry.get("stream") is not None:
        det._stream = det.detector.load_stream(entry["stream"])
    if entry["nulls"] is not None:
        det._nulls = [
            NullDistribution(samples=arrays[f"{prefix}_null{h}"], **params)
            for h, params in enumerate(entry["nulls"])
        ]


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
