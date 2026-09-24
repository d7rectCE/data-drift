# API reference

Generated from the docstrings by `python docs/gen_api.py`; do not edit by hand.
Everything listed under `driftfdr.__all__` can be imported from the top-level package.

Top-level exports: `ADWIN`, `AlphaInvesting`, `BHWindow`, `BatchBH`, `BonferroniWindow`, `CalibratedDetector`, `CalibrationConfig`, `DDM`, `Detector`, `EBHWindow`, `KSSliding`, `KSWindow`, `LOND`, `LORDpp`, `MeanShift`, `MonitorConfig`, `MonitorResult`, `NullDistribution`, `PageHinkley`, `RawThreshold`, `SAFFRON`, `Scenario`, `ScenarioConfig`, `StoreyBHWindow`, `StreamingMonitor`, `SupervisedConfig`, `Uncorrected`, `bucket_means`, `calibrate`, `calibrate_many`, `default_detectors`, `from_river`, `make_procedure`, `make_scenario`, `make_supervised_scenario`, `run_monitor`, `split_common`, `summarize`, `tolerance_from_cost`.

## Contents

- [`driftfdr.streaming`](#driftfdrstreaming)
- [`driftfdr.detectors`](#driftfdrdetectors)
- [`driftfdr.calibration`](#driftfdrcalibration)
- [`driftfdr.online_fdr`](#driftfdronline_fdr)
- [`driftfdr.preprocess`](#driftfdrpreprocess)
- [`driftfdr.bootstrap`](#driftfdrbootstrap)
- [`driftfdr.monitor`](#driftfdrmonitor)
- [`driftfdr.metrics`](#driftfdrmetrics)
- [`driftfdr.streams`](#driftfdrstreams)
- [`driftfdr.datasets`](#driftfdrdatasets)

## `driftfdr.streaming`

Streaming interface: feed observations one step at a time, get p-values and alarms.

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

### `CalibratedDetector`

```python
CalibratedDetector(detector: Detector, n_ref: int = 300, window: int = 100, horizon: int = 5, calibration: CalibrationConfig = CalibrationConfig(n_boot=2000, method='sieve_pu', block_length='auto', tail='gpd', tail_fraction=0.1, min_exceedances=10, seed=12345, tolerance=0.0), seed=None)
```

One detector for one model: collects a reference, calibrates, then emits p-values.

``update(x)`` returns ``None`` while the reference is being collected and between
window ends, and a p-value at the end of every window afterwards. The statistic
looks back over up to ``horizon`` windows; one bootstrap pass calibrates all of
them. ``reset()`` starts a new reference, e.g. after retraining.

- **`reset(seed=None)`** 

  Start collecting a new reference, e.g. right after retraining.

- **`calibrated`** *(property)* — Whether the reference is complete and the null distributions are ready.
- **`update(x: float) -> float | None`** 

  Add one observation; returns a p-value at the end of each window after calibration.

### `StreamingMonitor`

```python
StreamingMonitor(n_models: int | None = None, detector_factory=None, procedure: Procedure | str = 'bonferroni', alpha: float = 0.05, n_ref: int = 300, window: int = 100, horizon: int = 5, calibration: CalibrationConfig = CalibrationConfig(n_boot=2000, method='sieve_pu', block_length='auto', tail='gpd', tail_fraction=0.1, min_exceedances=10, seed=12345, tolerance=0.0), seed: int = 0, model_ids=None, split_common: bool = False)
```

One calibrated detector per model plus a rule that decides which models to retrain.

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

After every ``update``, ``last_pvalues`` holds the p-values that became ready in that
call, keyed by model id (and ``StreamingMonitor.FLEET`` for the common component).

- **`add_model(model_id) -> None`** 

  Start monitoring a new model; it first collects its reference.

- **`remove_model(model_id) -> None`** 

  Stop monitoring a model and forget its state.

- **`reset_model(model_id) -> None`** 

  Call after retraining a model for any reason (alarmed models are reset automatically).

- **`update(observations)`** 

  Feed new observations; return the models to retrain now.

  ``observations`` is a sequence with one value per model (models ``0..n-1``,
  returns an integer array) or a mapping ``{model_id: value}`` for any subset
  of models (returns a list of ids). Alarmed models are reset automatically.

- **`save(path) -> None`** 

  Write the full state to an ``.npz`` file (arrays plus JSON metadata, no pickle).

- **`load(path, detector_factory, procedure: Procedure | None = None) -> StreamingMonitor`** *(classmethod)* 

  Restore a monitor saved with ``save``. Custom procedures must be passed in again.

### `from_river`

```python
from_river(river_detector) -> Detector
```

Vectorised equivalent of a configured ``river.drift`` detector (PageHinkley, DDM, ADWIN).

## `driftfdr.detectors`

Drift detectors rewritten as continuous, vectorised scores.

A practical detector fires when an internal statistic crosses a threshold set
by its sensitivity hyperparameter (``lambda`` for Page-Hinkley, the number of
standard deviations for DDM, ``delta`` for ADWIN, ``alpha`` for KS). Each class
below exposes that statistic as a *score* on a common scale, so that

    detector with hyperparameter theta fires at time t  <=>  score_t > h(theta).

The score is the "critical sensitivity" at which the detector would have
fired, which is what makes it calibratable into a p-value: the binary alarm is
recovered by thresholding, and ``default_threshold`` reproduces the river
default. Page-Hinkley and DDM reproduce river's formulas exactly up to the
first alarm (see ``tests/test_detectors.py``); ADWIN uses river's cut criterion
on a fixed geometric grid of split points instead of the exponential histogram.

All ``scores`` methods take an array of shape ``(n_series, n_steps)`` and
return an array of the same shape, so thousands of bootstrap replicates are
scored in one call. Scores are causal: the score at step ``t`` only uses data
up to ``t``, so one pass over a long series yields the statistics of all its
prefixes.

### `Detector` (ABC)

```python
Detector()
```

Base class: a drift detector exposed as a continuous, vectorised score.

Subclasses implement ``scores`` (sequential detectors, whose statistic for a
window is the maximum score inside it) or override ``window_statistics``
directly (window tests such as KS or MeanShift), and give the score threshold
that reproduces the detector's usual default as ``default_threshold``.

- **`default_threshold`** *(property)* — Score threshold equivalent to the river default hyperparameters.
- **`scores(x: np.ndarray) -> np.ndarray`** 

  Running score of shape ``(n_series, n_steps)`` for input of the same shape.

  The score at step ``t`` depends only on data up to ``t``; the detector would
  fire at ``t`` with sensitivity ``theta`` iff the score exceeds ``h(theta)``.

- **`window_statistics(x: np.ndarray, n_ref: int, window: int) -> np.ndarray`** 

  Test statistics for "data after the reference differ from it".

  ``x`` holds the reference in its first ``n_ref`` columns followed by
  ``n`` windows of ``window`` steps. The detector is warmed up on the
  reference and run over the windows; column ``i`` of the result is the
  maximum score reached inside window ``i``, shape ``(n_series, n)``.

### `PageHinkley` (Detector)

```python
PageHinkley(delta: float = 0.005, alpha: float = 0.9999, min_instances: int = 30, mode: str = 'up', threshold: float = 50.0)
```

Page-Hinkley test as in ``river.drift.PageHinkley``; score is compared with ``lambda``.

- **`default_threshold`** *(property)* — Score threshold equivalent to the river default hyperparameters.
- **`scores(x: np.ndarray) -> np.ndarray`** 

  Running score of shape ``(n_series, n_steps)`` for input of the same shape.

  The score at step ``t`` depends only on data up to ``t``; the detector would
  fire at ``t`` with sensitivity ``theta`` iff the score exceeds ``h(theta)``.

### `DDM` (Detector)

```python
DDM(warm_start: int = 30, drift_threshold: float = 3.0)
```

Drift Detection Method as in ``river.drift.binary.DDM``.

Score is ``(p_t + s_t - p_min) / s_min``; river fires when it exceeds
``drift_threshold`` (3 by default).

- **`default_threshold`** *(property)* — Score threshold equivalent to the river default hyperparameters.
- **`scores(x: np.ndarray) -> np.ndarray`** 

  Running score of shape ``(n_series, n_steps)`` for input of the same shape.

  The score at step ``t`` depends only on data up to ``t``; the detector would
  fire at ``t`` with sensitivity ``theta`` iff the score exceeds ``h(theta)``.

### `ADWIN` (Detector)

```python
ADWIN(delta: float = 0.002, min_window_length: int = 5, grace_period: int = 10, n_splits: int = 24)
```

ADWIN-style statistic with river's cut criterion.

For a window ``W = W0 + W1`` of width ``n`` river cuts when
``|mu0 - mu1| > eps``, ``eps = sqrt(2 m v L) + 2/3 m L``, where
``L = ln(2 ln n / delta)``, ``v`` is the window variance and ``m`` the
harmonic term of the two sub-window sizes. Solving for the smallest
``delta`` that produces a cut gives the score ``-ln(delta*)``; river fires
when it exceeds ``-ln(delta)``. The window grows from the start of the
reference (no shrinking) and split points lie on a geometric grid.

- **`default_threshold`** *(property)* — Score threshold equivalent to the river default hyperparameters.
- **`scores(x: np.ndarray) -> np.ndarray`** 

  Running score of shape ``(n_series, n_steps)`` for input of the same shape.

  The score at step ``t`` depends only on data up to ``t``; the detector would
  fire at ``t`` with sensitivity ``theta`` iff the score exceeds ``h(theta)``.

### `KSWindow` (Detector)

```python
KSWindow(alpha: float = 0.05, n_ref: int = 300, window: int = 100)
```

Two-sample Kolmogorov–Smirnov statistic between the reference and recent data.

The statistic for window ``i`` compares the reference with everything from
the end of the reference up to the end of that window. The default
threshold is the asymptotic 5% critical value for a single window under
i.i.d. observations, i.e. what ``scipy.stats.ks_2samp`` would use.

- **`default_threshold`** *(property)* — Score threshold equivalent to the river default hyperparameters.
- **`nominal_pvalue(statistic: np.ndarray, n_ref: int, window: int) -> np.ndarray`** 

  Asymptotic KS p-value assuming i.i.d. data (what ``scipy.stats.ks_2samp`` reports).

- **`window_statistics(x: np.ndarray, n_ref: int, window: int) -> np.ndarray`** 

  Test statistics for "data after the reference differ from it".

  ``x`` holds the reference in its first ``n_ref`` columns followed by
  ``n`` windows of ``window`` steps. The detector is warmed up on the
  reference and run over the windows; column ``i`` of the result is the
  maximum score reached inside window ``i``, shape ``(n_series, n)``.

### `KSSliding` (Detector)

```python
KSSliding(width: int = 100, stride: int = 10, alpha: float = 0.05, n_ref: int = 300)
```

Sequential KS in the spirit of IKS (dos Reis et al., 2016).

The reference is compared with a sliding window of the last ``width``
observations every ``stride`` steps; the statistic of a check window is the
largest KS distance reached inside it. Unlike ``KSWindow`` it forgets old
data, so a short-lived change is not diluted by a long look-back.

- **`default_threshold`** *(property)* — Score threshold equivalent to the river default hyperparameters.
- **`window_statistics(x: np.ndarray, n_ref: int, window: int) -> np.ndarray`** 

  Test statistics for "data after the reference differ from it".

  ``x`` holds the reference in its first ``n_ref`` columns followed by
  ``n`` windows of ``window`` steps. The detector is warmed up on the
  reference and run over the windows; column ``i`` of the result is the
  maximum score reached inside window ``i``, shape ``(n_series, n)``.

### `MeanShift` (Detector)

```python
MeanShift(persistence: int = 3)
```

Persistent rise of the mean: the smallest window-mean excess over the last ``persistence`` windows.

The statistic for window ``i`` is ``min_{j in last m windows} mean(window j) - mean(reference)``
where windows not yet observed count as zero excess, so no alarm is possible before
``persistence`` windows have passed. A spike confined to one window cannot make it
large when ``persistence > 1``, so alarms require the rise to last. With
``persistence = 1`` it is the plain window-vs-reference mean test used by Rombouts &
Wilms for loss monitoring. There is no river counterpart; ``default_threshold`` is
a fixed rise of 0.1 in signal units.

- **`default_threshold`** *(property)* — Score threshold equivalent to the river default hyperparameters.
- **`window_statistics(x: np.ndarray, n_ref: int, window: int) -> np.ndarray`** 

  Test statistics for "data after the reference differ from it".

  ``x`` holds the reference in its first ``n_ref`` columns followed by
  ``n`` windows of ``window`` steps. The detector is warmed up on the
  reference and run over the windows; column ``i`` of the result is the
  maximum score reached inside window ``i``, shape ``(n_series, n)``.

### `ks_statistic`

```python
ks_statistic(a: np.ndarray, b: np.ndarray) -> np.ndarray
```

Row-wise two-sample KS statistic, correct in the presence of ties.

### `default_detectors`

```python
default_detectors(n_ref: int = 300, window: int = 100) -> list[Detector]
```

The four detectors of the original study, with river-default thresholds: PH, DDM, ADWIN, KS.

## `driftfdr.calibration`

Turning detector statistics into p-values by resampling.

For a stream with reference segment ``R`` the null distribution of the window
statistic is estimated by scoring ``B`` bootstrap series resampled from ``R``:
a pseudo-reference of ``len(R)`` steps followed by a pseudo-window. Both parts
are resampled, so their mutual variability matches that of a fresh window
compared with the observed reference. The p-value is the usual
``(1 + #{T* >= T}) / (B + 1)``.

With ``B`` in the hundreds the smallest attainable p-value is ``1 / (B + 1)``,
far above the per-test levels online FDR procedures reach with hundreds of
streams. When fewer than ``min_exceedances`` bootstrap statistics exceed the
observed one, the p-value is extrapolated from a tail model fitted to the top
``tail_fraction`` of the bootstrap sample (Knijnenburg et al., 2009). The
default tail is exponential: the statistics here are maxima of sums or
squared standardised differences, which lie in the Gumbel domain, and fixing
the shape avoids the large variance of an estimated GPD shape. A GPD fit by
probability-weighted moments is available as ``tail="gpd"``.

### `CalibrationConfig`

```python
CalibrationConfig(n_boot: int = 2000, method: str = 'sieve_pu', block_length: int | str = 'auto', tail: str = 'gpd', tail_fraction: float = 0.1, min_exceedances: int = 10, seed: int = 12345, tolerance: float = 0.0) -> None
```

Defaults follow the experiments: AR-sieve with parameter uncertainty (moving
blocks are used automatically for 0/1 signals), 2000 replicates and a GPD tail,
which bring the per-window FWER of Bonferroni to its nominal level for
Page-Hinkley and KS (experiment 10). Experiments 1-9 and 11-15 pin the earlier
defaults (500 replicates, exponential tail) explicitly.

| field | type | default | description |
|---|---|---|---|
| `n_boot` | `int` | `2000` |  |
| `method` | `str` | `'sieve_pu'` |  |
| `block_length` | `int | str` | `'auto'` |  |
| `tail` | `str` | `'gpd'` | Tail model for small p-values: ``exponential``, ``gpd`` or ``none``. |
| `tail_fraction` | `float` | `0.1` |  |
| `min_exceedances` | `int` | `10` |  |
| `seed` | `int` | `12345` |  |
| `tolerance` | `float` | `0.0` | Null of *material* change: the new data may exceed the reference level by up to this much (signal units). The bootstrap continuation is shifted up by it, the least favourable point of that null, so p-values are valid for every smaller increase. |

### `NullDistribution`

```python
NullDistribution(samples: np.ndarray, block_length: int, min_exceedances: int = 10, tail_threshold: float = nan, tail_scale: float = nan, tail_shape: float = nan, tail_prob: float = nan) -> None
```

Bootstrap null distribution of a window statistic, with an optional fitted upper tail.

| field | type | default | description |
|---|---|---|---|
| `samples` | `np.ndarray` | `required` | Sorted bootstrap statistics (may contain ``inf``). |
| `block_length` | `int` | `required` | Block length, or the AR order for the sieve bootstrap. |
| `min_exceedances` | `int` | `10` |  |
| `tail_threshold` | `float` | `nan` |  |
| `tail_scale` | `float` | `nan` |  |
| `tail_shape` | `float` | `nan` |  |
| `tail_prob` | `float` | `nan` |  |

- **`from_samples(samples: np.ndarray, block_length: int, config: CalibrationConfig) -> NullDistribution`** *(classmethod)* 

  Sort the bootstrap statistics and fit the tail model requested by ``config``.

- **`has_tail`** *(property)* — Whether a tail model was fitted (it is not for degenerate or tiny samples).
- **`pvalue(statistic) -> np.ndarray`** 

  p-values of one or more observed statistics, as an array.

  ``(1 + #{T* >= T}) / (B + 1)``, replaced by the tail model when fewer than
  ``min_exceedances`` bootstrap statistics reach ``T``.

### `fit_tail`

```python
fit_tail(sorted_samples: np.ndarray, tail_fraction: float, model: str = 'exponential', min_tail: int = 30)
```

Fit an exponential or GPD tail to the largest bootstrap statistics.

The GPD is fitted by probability-weighted moments (Hosking & Wallis, 1987)
with the shape clipped to ``[0, 0.9]``. Returns
``(threshold, scale, shape, P(T > threshold))`` or ``None``.

### `resolve_block_length`

```python
resolve_block_length(reference: np.ndarray, method: str, config: CalibrationConfig) -> int
```

Block length for a block method: 1 for iid, the AR(1) plug-in for ``"auto"``, else the given value.

### `bootstrap_series`

```python
bootstrap_series(reference, length: int, config: CalibrationConfig, rng, binary: bool = False)
```

``B`` pseudo series: a resampled reference followed by ``length`` resampled steps.

With ``config.tolerance > 0`` the continuation is raised by the tolerance: added to
continuous values, or, for binary errors, by turning zeros into ones with the
probability that raises the error rate by that much.

### `calibrate_many`

```python
calibrate_many(detector: Detector, references: np.ndarray, window: int, horizon: int, config: CalibrationConfig, seeds, max_chunk_elements: int = 4000000) -> list[list[NullDistribution]]
```

Bootstrap null distributions for several streams at once.

``references`` has shape ``(n_streams, n_ref)``. For every stream the
result holds ``horizon`` null distributions: entry ``h - 1`` is for the
statistic of the last window when the detector has seen ``h`` windows
since the reference. Because scores are causal, all of them come from a
single pass over bootstrap series of ``n_ref + horizon * window`` steps.
``seeds`` gives one seed (anything ``np.random.default_rng`` accepts) per
stream, so results do not depend on which streams are calibrated together.

### `calibrate`

```python
calibrate(detector: Detector, reference, window: int, horizon: int = 1, config=CalibrationConfig(n_boot=2000, method='sieve_pu', block_length='auto', tail='gpd', tail_fraction=0.1, min_exceedances=10, seed=12345, tolerance=0.0), seed=0) -> list[NullDistribution]
```

Null distributions for one stream, one per number of windows seen.

## `driftfdr.online_fdr`

Decision rules applied to the p-values of one monitoring window.

Every window produces a batch of p-values, one per actively monitored stream.
Per-window rules (``Uncorrected``, ``BonferroniWindow``, ``BHWindow``,
``StoreyBHWindow``) decide on each batch on its own. ``BatchBH`` runs BH inside
each batch at levels chosen to control FDR over all batches. Online rules
(``LOND``, ``LORDpp``, ``SAFFRON``, ``AlphaInvesting``) see hypotheses one at a
time; within a batch they are fed in a random order that does not depend on
the p-values.

References: Foster & Stine (2008), Storey, Taylor & Siegmund (2004),
Javanmard & Montanari (2018), Ramdas et al. (2017, LORD++), Ramdas et al.
(2018, SAFFRON), Zrnic, Jiang, Ramdas & Jordan (2020, BatchBH).

### `gamma_lord`

```python
gamma_lord(j: np.ndarray) -> np.ndarray
```

Javanmard–Montanari sequence, sums to one over j >= 1.

### `gamma_saffron`

```python
gamma_saffron(j: np.ndarray) -> np.ndarray
```

``gamma_j ∝ j^-1.6``, sums to one over j >= 1.

### `Procedure` (ABC)

```python
Procedure(alpha: float = 0.05)
```

Base class of decision rules: given the p-values of one window, which models alarm.

- **`decide(values: np.ndarray, rng) -> np.ndarray`** 

  Boolean rejections for one window.

### `RawThreshold` (Procedure)

```python
RawThreshold(threshold: float)
```

Uncalibrated detector: alarm when the statistic crosses the default threshold.

- **`decide(values, rng)`** 

  Boolean rejections for one window.

### `Uncorrected` (Procedure)

```python
Uncorrected(alpha: float = 0.05)
```

Each stream tested at level alpha, as in current practice.

- **`decide(values, rng)`** 

  Boolean rejections for one window.

### `BonferroniWindow` (Procedure)

```python
BonferroniWindow(alpha: float = 0.05)
```

Bonferroni within each window: controls P(any false alarm in a window).

- **`decide(values, rng)`** 

  Boolean rejections for one window.

### `BHWindow` (Procedure)

```python
BHWindow(alpha: float = 0.05)
```

Benjamini–Hochberg within each window.

- **`decide(values, rng)`** 

  Boolean rejections for one window.

### `StoreyBHWindow` (Procedure)

```python
StoreyBHWindow(alpha: float = 0.05, lam: float = 0.5)
```

Adaptive BH within each window: BH at ``alpha / pi0_hat``.

``pi0_hat = (1 + #{p > lam}) / (m (1 - lam))`` is the finite-sample Storey
estimate of the fraction of nulls; when many streams drift at once it drops
and the threshold rises.

- **`decide(values, rng)`** 

  Boolean rejections for one window.

### `EBHWindow` (Procedure)

```python
EBHWindow(alpha: float = 0.05, kappa: float = 0.5)
```

e-BH within each window (Wang & Ramdas, 2022): FDR control under any dependence.

p-values are turned into e-values by the calibrator ``e = kappa * p^(kappa - 1)``,
which integrates to one over a uniform p, and the ``k`` largest e-values are
rejected with ``k = max{k : e_(k) >= m / (alpha k)}``. Unlike BH it needs no
assumption on how the streams depend on each other, at a price in power.

- **`decide(values, rng)`** 

  Boolean rejections for one window.

### `BatchBH` (Procedure)

```python
BatchBH(alpha: float = 0.05)
```

BH inside each window at levels that control FDR across all windows.

With ``R_s`` rejections and ``R_s^+`` the rejections BH would make in batch
``s`` if one of its p-values were set to zero, the level of batch ``t`` is

    (alpha * sum_{s<=t} gamma_s - sum_{s<t} alpha_s R_s^+ / (R_s^+ + sum_{r<s} R_r))
    * (n_t + sum_{r<t} R_r) / n_t,

which keeps ``sum_s alpha_s R_s^+ / (R_s^+ + sum_{r<s} R_r) <= alpha``, the
bound on FDR for independent p-values. Past rejections both refund budget
and scale the level up.

- **`decide(values, rng)`** 

  Boolean rejections for one window.

### `bh_count`

```python
bh_count(p: np.ndarray, alpha: float) -> int
```

Number of BH rejections at level ``alpha``.

### `benjamini_hochberg`

```python
benjamini_hochberg(p: np.ndarray, alpha: float) -> np.ndarray
```

Boolean rejections of the Benjamini–Hochberg step-up procedure at level ``alpha``.

### `OnlineProcedure` (Procedure)

```python
OnlineProcedure(alpha: float = 0.05)
```

Sequential rule: hypotheses ``t = 1, 2, ...`` each get a level ``alpha_t``.

- **`decide(values, rng)`** 

  Boolean rejections for one window.

- **`test(p: float) -> bool`** 

  Test the next hypothesis: compute its level, decide, update the state.

- **`next_level() -> float`** 

  Level ``alpha_t`` of the hypothesis about to be tested.

- **`update(p: float, rejected: bool) -> None`** 

  Record the outcome of the hypothesis just tested.

### `LOND` (OnlineProcedure)

```python
LOND(alpha: float = 0.05)
```

``alpha_t = alpha * gamma_t * (D_{t-1} + 1)``; FDR control under PRDS.

- **`next_level()`** 

  Level ``alpha_t`` of the hypothesis about to be tested.

- **`update(p, rejected)`** 

  Record the outcome of the hypothesis just tested.

### `LORDpp` (OnlineProcedure)

```python
LORDpp(alpha: float = 0.05, w0: float | None = None)
```

LORD++: wealth is earned back at every rejection.

- **`next_level()`** 

  Level ``alpha_t`` of the hypothesis about to be tested.

- **`update(p, rejected)`** 

  Record the outcome of the hypothesis just tested.

### `SAFFRON` (OnlineProcedure)

```python
SAFFRON(alpha: float = 0.05, lam: float = 0.5, w0: float | None = None)
```

SAFFRON: adapts to the fraction of nulls through candidates ``p <= lambda``.

- **`next_level()`** 

  Level ``alpha_t`` of the hypothesis about to be tested.

- **`update(p, rejected)`** 

  Record the outcome of the hypothesis just tested.

### `AlphaInvesting` (OnlineProcedure)

```python
AlphaInvesting(alpha: float = 0.05, w0: float | None = None, payout: float | None = None)
```

Foster–Stine alpha-investing (mFDR control) with the ``W / (1 + t - k*)`` spending rule.

- **`next_level()`** 

  Level ``alpha_t`` of the hypothesis about to be tested.

- **`update(p, rejected)`** 

  Record the outcome of the hypothesis just tested.

### `make_procedure`

```python
make_procedure(name: str, alpha: float = 0.05) -> Procedure
```

A fresh procedure by name: one of the keys of ``PROCEDURES``.

## `driftfdr.preprocess`

Turning raw per-event errors into monitorable series, and choosing the tolerance.

Monitoring works on one value per model per step. Real systems log errors at
irregular times (many predictions in a busy hour, none at night), and many
signals have daily or weekly cycles. ``bucket_means`` averages events into
fixed time buckets, so a step is an hour or a day rather than a row; with a
window of one full cycle, the cycle itself cancels out (see experiment 16).

### `bucket_means`

```python
bucket_means(values: np.ndarray, bucket: np.ndarray, n_buckets: int | None = None) -> np.ndarray
```

Mean of ``values`` (shape ``(n_events,)`` or ``(n_series, n_events)``) per integer bucket id.

Empty buckets are filled with the previous bucket's mean (the first one with the
overall mean), so the output has no gaps; returns shape ``(n_series, n_buckets)``.

### `tolerance_from_cost`

```python
tolerance_from_cost(retrain_cost: float, horizon: float) -> float
```

Smallest rise of the per-step loss that is worth one retraining.

A drifted model left alone costs ``delta`` extra loss per step for ``horizon``
steps (until the next scheduled retraining, or the planning horizon); a
retraining costs ``retrain_cost`` in the same loss units. Retraining pays off
when ``delta * horizon > retrain_cost``, so the tolerance of the
material-degradation null is ``retrain_cost / horizon``.

### `split_common`

```python
split_common(values: np.ndarray, n_ref: int) -> tuple[np.ndarray, np.ndarray]
```

Split synchronous streams into a common component and model-specific residuals.

Each stream is standardised by the mean and standard deviation of its first
``n_ref`` steps; the common component is the cross-sectional median at every
step, and a residual is a stream minus it. Fluctuations shared by all models
(the source of bursts of false alarms, experiment 5) end up in the common
component, which is monitored as one extra stream; a drift that hits a minority
of models barely moves the median and stays in their residuals. Returns
``(residuals of shape (n_streams, n_steps), common of shape (n_steps,))``.

## `driftfdr.bootstrap`

Resampling schemes for autocorrelated series.

* ``moving``: moving block bootstrap (Künsch, 1989), fixed block length;
* ``stationary``: stationary bootstrap (Politis & Romano, 1994), geometric
  block lengths with the given mean;
* ``sieve``: AR-sieve bootstrap (Bühlmann, 1997), an AR(p) fitted by
  Yule–Walker with AIC order selection, driven by resampled residuals; only
  meaningful for continuous signals;
* ``sieve_pu``: AR-sieve with parameter uncertainty. Each replicate refits the
  AR model on a series simulated from the fitted one and is generated from its
  own refitted coefficients, so the estimation error of the reference model
  (the source of the anti-conservative tails the plain sieve shows) is carried
  into the null distribution. Same idea as the correction of Wu & Apley for
  nested bootstraps.
* ``iid``: ordinary bootstrap, kept as a baseline that ignores dependence.

### `ar1_block_length`

```python
ar1_block_length(x: np.ndarray, method: str = 'moving') -> int
```

Plug-in block length for an AR(1) approximation of ``x``.

Uses the Politis–White (2004) optimal-rate formula specialised to AR(1):
``b = c * (2 r / (1 - r^2))^(2/3) * n^(1/3)``, with ``c = (3/2)^(1/3)``
for block bootstrap and ``c = 1`` for the stationary bootstrap, where ``r``
is the lag-1 autocorrelation.

### `bootstrap_indices`

```python
bootstrap_indices(n_source: int, n_out: int, n_boot: int, block_length: int, method: str, rng) -> np.ndarray
```

Indices into a source series of length ``n_source``, shape ``(n_boot, n_out)``.

### `fit_ar_aic`

```python
fit_ar_aic(x: np.ndarray, max_order: int | None = None) -> np.ndarray
```

Yule–Walker AR coefficients ``a`` (``x_t = sum_i a_i x_{t-i} + e_t``), order by AIC.

### `ar_sieve_series`

```python
ar_sieve_series(x: np.ndarray, n_out: int, n_boot: int, rng, burn_in: int = 200)
```

AR-sieve bootstrap replicates of ``x``; returns ``(series, order)``.

### `ar_sieve_pu_series`

```python
ar_sieve_pu_series(x: np.ndarray, n_out: int, n_boot: int, rng, burn_in: int = 200)
```

AR-sieve replicates whose coefficients are redrawn per replicate; returns ``(series, order)``.

Coefficients of replicate ``b`` are the Yule–Walker fit (same order as the
original AIC choice) to a series of ``len(x)`` steps simulated from the
model fitted to ``x``, i.e. a parametric bootstrap draw of the estimator.

## `driftfdr.monitor`

Multi-stream monitoring loop with retraining on alarm.

Time is split into check windows of ``window`` steps. Every stream owns a
reference segment of ``n_ref`` steps collected right after its last
(re)training. At the end of each window, every stream whose reference is
complete is tested: its detector is warmed up on the reference and run over
the most recent ``min(elapsed, horizon)`` windows, and the statistic is the
maximum score inside the current window. The look-back lets evidence
accumulate for up to ``horizon`` windows, as it does for a detector running
continuously in production, while keeping one calibration per reference.

The procedure then decides which streams alarm; alarmed streams are retrained
and their new reference is the next ``n_ref`` steps, during which they are
not monitored.

### `MonitorConfig`

```python
MonitorConfig(n_ref: int = 300, window: int = 100, horizon: int = 5, calibration: CalibrationConfig = <factory>) -> None
```

Reference length, window length and look-back (in windows) of the monitoring loop.

| field | type | default | description |
|---|---|---|---|
| `n_ref` | `int` | `300` |  |
| `window` | `int` | `100` |  |
| `horizon` | `int` | `5` | Look-back in windows; 1 compares each window with the reference alone. |
| `calibration` | `CalibrationConfig` | `'CalibrationConfig()'` |  |

### `MonitorResult`

```python
MonitorResult(tests: pd.DataFrame, n_windows: int, scenario: Scenario, config: MonitorConfig) -> None
```

Outcome of ``run_monitor``: every test with its decision and ground truth.

| field | type | default | description |
|---|---|---|---|
| `tests` | `pd.DataFrame` | `required` | One row per (stream, window) test: statistic, p-value, decision, ground truth. |
| `n_windows` | `int` | `required` |  |
| `scenario` | `Scenario` | `required` |  |
| `config` | `MonitorConfig` | `required` |  |

### `run_monitor`

```python
run_monitor(scenario: Scenario, detector: Detector, procedure: Procedure | None, config: MonitorConfig = MonitorConfig(n_ref=300, window=100, horizon=5, calibration=CalibrationConfig(n_boot=2000, method='sieve_pu', block_length='auto', tail='gpd', tail_fraction=0.1, min_exceedances=10, seed=12345, tolerance=0.0)), seed: int = 0, cache: NullCache | None = None) -> MonitorResult
```

Run one monitoring pass.

``procedure=None`` never alarms and only records statistics and p-values,
which is how null calibration is checked. ``cache`` maps
``(stream, reference_start)`` to calibrated null distributions; pass the
same dict to runs on the same scenario, detector and config (e.g. different
procedures) to reuse calibrations.

## `driftfdr.metrics`

Evaluation of a monitoring run against the known change points.

Error side: FDP (false alarms / alarms over the whole run), the per-window
FDP averaged over windows (what a per-window BH controls), false alarms per
window, probability of at least one false alarm in a window, and MTFA in
stream-steps.
Combined: MTR = MTFA / MTD * (1 - MDR) (Bifet et al., 2013), where MTFA is in
stream-steps per false alarm, so it is comparable across fleets of equal size.
Detection side: for every drifting stream the delay is the time from the onset
to the end of the first window whose test alarms; a drift that is never
caught counts as missed (MDR) and keeps degrading the model until the end of
the run. ``degraded_per_drift`` averages this censored delay over all drifts,
so it penalises both slow detection and misses.
Event-level precision, recall and F1 as in detector benchmarks (Cerqueira et
al., 2026): an alarm is a true detection if it is the first alarm after a
change (within ``max_delay`` steps, if given, and before the next change);
every other alarm, including repeated alarms for a change already caught, is
a false detection. Unlike ``fdp``, this does not use the null hypothesis of
the tests, so it is comparable with published benchmarks.

### `summarize`

```python
summarize(result: MonitorResult, max_delay: int | None = None) -> dict
```

Error and detection metrics of one monitoring run, as a flat dict (see the module docstring).

``max_delay`` only affects the event-level ``precision``, ``recall`` and ``f1``: a
detection later than this many steps after the change does not count.

## `driftfdr.streams`

Synthetic multi-stream scenarios with known change points.

Each stream stands for the monitored signal of one production model, e.g. its
per-step loss. Streams are AR(1) in time and share a common factor, so both
serial and cross-stream dependence are present. A fraction of the streams
undergoes a single change in mean, abrupt or gradual, at a known time.
Changes are either sporadic (each stream at its own time) or clustered into
drift events that shift a whole group of streams at the same moment, as when
an upstream data source changes for many models at once.

Two views of the same latent process are exposed:

* ``values``: a continuous signal (loss proxy), used by Page-Hinkley, ADWIN, KS;
* ``errors``: binary error indicators ``1{latent > c}`` with base rate
  ``base_error_rate``, used by DDM.

### `ScenarioConfig`

```python
ScenarioConfig(n_streams: int = 100, n_steps: int = 5000, phi: float = 0.5, rho: float = 0.3, drift_fraction: float = 0.1, drift_type: str = 'abrupt', magnitude: float = 1.0, magnitude_range: tuple[float, float] | None = None, gradual_length: int = 300, onset_range: tuple[float, float] = (0.3, 0.8), fixed_onset: int | None = None, drift_events: int = 0, event_fraction: float = 0.1, base_error_rate: float = 0.2) -> None
```

Parameters of a synthetic scenario of AR(1) model-error streams with known changes.

| field | type | default | description |
|---|---|---|---|
| `n_streams` | `int` | `100` |  |
| `n_steps` | `int` | `5000` |  |
| `phi` | `float` | `0.5` | AR(1) coefficient of every stream. |
| `rho` | `float` | `0.3` | Contemporaneous correlation between any two streams (common factor). |
| `drift_fraction` | `float` | `0.1` |  |
| `drift_type` | `str` | `'abrupt'` |  |
| `magnitude` | `float` | `1.0` | Size of the mean shift, in units of the marginal standard deviation. |
| `magnitude_range` | `tuple[float, float] | None` | `None` | If set, each drifting stream draws its shift uniformly from this range instead. |
| `gradual_length` | `int` | `300` |  |
| `onset_range` | `tuple[float, float]` | `(0.3, 0.8)` | Onsets are drawn uniformly from this fraction range of ``n_steps``. |
| `fixed_onset` | `int | None` | `None` | If set, every drifting stream changes at this step instead. |
| `drift_events` | `int` | `0` | Number of drift events; each shifts ``event_fraction`` of the streams at once. |
| `event_fraction` | `float` | `0.1` |  |
| `base_error_rate` | `float` | `0.2` |  |

### `Scenario`

```python
Scenario(config: ScenarioConfig, values: np.ndarray, errors: np.ndarray, change_start: np.ndarray, change_end: np.ndarray, drift_kind: np.ndarray, event: np.ndarray, later_changes: np.ndarray | None = None, truth: np.ndarray | None = None, tolerance: float = 0.0, truth_ref: np.ndarray | None = None, mean_shift: np.ndarray | None = None, features: np.ndarray | None = None) -> None
```

Monitored signals of every stream plus the ground truth needed to score alarms.

| field | type | default | description |
|---|---|---|---|
| `config` | `ScenarioConfig` | `required` |  |
| `values` | `np.ndarray` | `required` |  |
| `errors` | `np.ndarray` | `required` |  |
| `change_start` | `np.ndarray` | `required` | First index at which the mean differs from its initial level. |
| `change_end` | `np.ndarray` | `required` | First index from which the mean stays at its final level. |
| `drift_kind` | `np.ndarray` | `required` |  |
| `event` | `np.ndarray` | `required` | Drift event of each stream, -1 for sporadic changes and stable streams. |
| `later_changes` | `np.ndarray | None` | `None` | Optional ``(n_streams, m, 2)`` array of further (start, end) changes after the first. |
| `truth` | `np.ndarray | None` | `None` | Optional ``(n_streams, n_steps)`` oracle level of the monitored quantity (e.g. the smoothed true error rate). When set, a test is null iff the level in the tested window exceeds the level over the reference by at most ``tolerance``: the null of *material degradation* instead of *any change*. |
| `tolerance` | `float` | `0.0` |  |
| `truth_ref` | `np.ndarray | None` | `None` | Optional level used for the reference side of the oracle comparison (defaults to ``truth``). |
| `mean_shift` | `np.ndarray | None` | `None` | True mean shift of every stream (synthetic scenarios only). |
| `features` | `np.ndarray | None` | `None` | Model input of every stream, for detectors that watch p(X) (supervised scenarios). |

- **`with_material_null(tolerance: float, truth: np.ndarray | None = None, truth_ref: np.ndarray | None = None) -> Scenario`** 

  Copy whose ground truth is *material degradation*: level up by more than ``tolerance``.

  ``truth`` defaults to the true mean shift of a synthetic scenario.

- **`changes() -> tuple[np.ndarray, np.ndarray]`** 

  All change starts and ends, shape ``(n_streams, n_changes)``, padded with ``NO_CHANGE``.

- **`n_streams`** *(property)* — Number of streams (models).
- **`n_steps`** *(property)* — Length of every stream.
- **`drifting`** *(property)* — Indices of streams with at least one change.
- **`signal(kind: str) -> np.ndarray`** 

  The array a detector consumes: ``"values"``, ``"errors"`` or ``"features"``.

- **`is_null(streams, start, stop, ref_len=None, window=None) -> np.ndarray`** 

  Ground truth of the test comparing the reference at ``start`` with the window ending at ``stop``.

  Regime null (default): no change anywhere between the start of the
  reference and the end of the window. With ``truth`` set: the oracle level
  in the last ``window`` steps exceeds its mean over the ``ref_len``-step
  reference by at most ``tolerance``.

### `ar1_latent`

```python
ar1_latent(n_streams: int, n_steps: int, phi: float, rho: float, rng) -> np.ndarray
```

Stationary unit-variance AR(1) streams with cross-correlation ``rho``.

### `make_scenario`

```python
make_scenario(config: ScenarioConfig, seed: int = 0) -> Scenario
```

Generate a synthetic scenario; the same config and seed always give the same data.

### `SupervisedConfig`

```python
SupervisedConfig(n_streams: int = 100, n_steps: int = 5000, phi: float = 0.5, rho: float = 0.0, drift_fraction: float = 0.6, kinds: tuple[str, ...] = ('virtual', 'real', 'both'), feature_shift: float = 1.0, slope_change: float = 0.5, period: int = 1000, onset_range: tuple[float, float] = (0.3, 0.6), error_quantile: float = 0.8) -> None
```

Streams of a feature, a label and a fixed linear model ``y_hat = beta0 * x``.

``virtual`` drift shifts the mean of x (p(X) changes, the model stays right);
``real`` drift changes the slope (p(y|X) changes, the error rises);
``both`` does both at once; ``cyclic`` switches the slope back and forth every
``period`` steps after the onset (a recurring concept).

| field | type | default | description |
|---|---|---|---|
| `n_streams` | `int` | `100` |  |
| `n_steps` | `int` | `5000` |  |
| `phi` | `float` | `0.5` |  |
| `rho` | `float` | `0.0` |  |
| `drift_fraction` | `float` | `0.6` |  |
| `kinds` | `tuple[str, ...]` | `('virtual', 'real', 'both')` |  |
| `feature_shift` | `float` | `1.0` | Shift of the feature mean for virtual drift, in feature standard deviations. |
| `slope_change` | `float` | `0.5` | Change of the slope for real drift (the model slope is 1, noise sd is 1). |
| `period` | `int` | `1000` |  |
| `onset_range` | `tuple[float, float]` | `(0.3, 0.6)` |  |
| `error_quantile` | `float` | `0.8` | 0/1 errors mark squared residuals above this quantile of the pre-drift residual. |

### `make_supervised_scenario`

```python
make_supervised_scenario(config: SupervisedConfig, seed: int = 0) -> Scenario
```

Scenario whose ``values`` are squared residuals of the model and ``features`` the input x.

Ground truth: ``mean_shift`` holds the true expected loss increase, so
``with_material_null`` judges alarms by model degradation; change points mark
every change of p(X) or p(y|X) for the regime null.

## `driftfdr.datasets`

Model-error streams from real data sets, in their original time order.

A "fleet" of K models is built on one data set: every model is a multinomial
logistic regression on its own random subset of features, trained once on an
initial segment. Its monitored signals are the per-instance log-loss
(``values``) and 0/1 error (``errors``). Unlike the benchmark of Cerqueira et
al., rows are never shuffled, so autocorrelation, seasonality and slow
wandering of the data stay in the streams.

* ``insects_scenario``: INSECTS (Souza et al., 2020) with its documented
  change points, shared by all models: real, clustered drifts. In the river
  copy each concept ends with a long run of a single class (class 5, rare in
  training), which every model misclassifies. ``ground_truth="extended"``
  adds the starts of such runs, found from the labels alone, as changes.
* ``elec2_scenario``: Electricity (Harries, 1999) with no labelled drifts.
  For a chosen fraction of models the labels they are scored against are
  flipped with some probability after a known onset, which is a real drift in
  p(y | X) injected into real temporal structure.

Ground truth is regime based: a test is null iff no documented or injected
change lies between the start of its reference and the end of its window.
Undocumented natural changes (INSECTS has visible ones, Electricity has
seasons) therefore count against the detectors; this is the null-definition
problem of real data made explicit.

Data are downloaded by ``river.datasets`` on first use.

### `fit_softmax`

```python
fit_softmax(X: np.ndarray, y: np.ndarray, n_classes: int, l2: float = 0.001, n_iter: int = 300, lr: float = 0.5)
```

Multinomial logistic regression by full-batch gradient descent on standardised features.

### `model_fleet`

```python
model_fleet(X, y, n_models, n_features, train_size, rng, label_noise=None)
```

Losses and errors of ``n_models`` models on random feature subsets, shape ``(n_models, n)``.

### `label_runs`

```python
label_runs(y: np.ndarray, min_length: int = 100) -> list[tuple[int, int]]
```

``(start, end)`` of runs of one repeated label at least ``min_length`` long.

### `insects_scenario`

```python
insects_scenario(n_models=50, n_features=8, train_size=3000, variant='abrupt_balanced', ground_truth='extended', seed=0) -> Scenario
```

INSECTS (Souza et al., 2020) with a fleet of models, monitored after the training segment.

``ground_truth="documented"`` uses the published change points only;
``"extended"`` adds the starts of long single-class runs found from the labels.
Needs river (the data are downloaded on first use).

### `elec2_scenario`

```python
elec2_scenario(n_models=50, n_features=4, train_size=5000, drift_fraction=0.2, flip=0.3, drift_events=0, seed=0) -> Scenario
```

Electricity with label-flip drift injected into ``drift_fraction`` of the models.

With ``drift_events > 0`` the drifting models are split into that many groups
sharing an onset (clustered drift); otherwise every drifting model gets its own.

### `forward_error`

```python
forward_error(errors: np.ndarray, span: int = 1000) -> np.ndarray
```

Error rate over the next ``span`` steps from each step (shorter at the end): what
the model will actually cost if it is not retrained now, known in hindsight.

### `error_rate_view`

```python
error_rate_view(scenario: Scenario, tolerance: float, span: int = 1000) -> Scenario
```

Monitor the 0/1 error stream; judge alarms by *material and persistent* degradation.

A test is null iff the error rate over the ``span`` steps following the tested
window's start exceeds the observed error rate over the reference by at most
``tolerance``. Transient spikes that pass by themselves are therefore null:
retraining for them would be wasted.

### `covertype_scenario`

```python
covertype_scenario(n_models=50, n_features=10, train_size=5000, n_rows=100000, seed=0) -> Scenario
```

Forest Covertype (Blackard, 1998) in its original order, no drift labels.

Needs scikit-learn (``fetch_covtype`` downloads the data on first use).

### `airlines_scenario`

```python
airlines_scenario(n_models=50, n_features=6, train_size=5000, n_rows=100000, seed=0) -> Scenario
```

Airlines delay data (OpenML 1169) in time order, no drift labels.

Numeric columns plus a one-hot airline code; the high-cardinality airport codes
are dropped. Needs scikit-learn (``fetch_openml`` downloads the data on first use).

### `airlines_hourly_scenario`

```python
airlines_hourly_scenario(n_models=50, n_features=6, seed=0) -> Scenario
```

All 31 days of Airlines, 0/1 delay errors averaged per (day, hour) bucket.

Models are trained on the first day. A step is one hour, so a window of 24
steps covers a full daily cycle. Days are counted from changes of DayOfWeek.
