# How the method works

**English** · [Русский](method.ru.md)

## 1. The detector as a continuous score (`detectors.py`)

Every practical detector has one sensitivity hyperparameter: λ for Page-Hinkley, the number of
sigmas for DDM, δ for ADWIN, α for KS. The detector fires exactly when an internal statistic
crosses the threshold set by that hyperparameter. The binary signal is therefore a thresholded
version of a continuous *score*, and the score equals "the least sensitivity at which the
detector would have fired":

| Detector | Score | river's default threshold |
|---|---|---|
| Page-Hinkley | `m_t − min m_s`, the cumulative deviation from the running mean | λ = 50 |
| DDM | `(p_t + s_t − p_min) / s_min` | 3 |
| ADWIN | `−ln δ*`, where δ* is the smallest δ at which the window has a cut | `−ln 0.002` |
| KS | Kolmogorov–Smirnov statistic, reference segment vs. the new data | critical value at α = 0.05 |
| Sliding KS (`KSSliding`, in the spirit of IKS) | largest KS distance between the reference and a sliding window of the latest observations | — |
| MeanShift(m) | minimum over the last m windows of the excess of the window mean over the reference mean (in signal units) | a rise of 0.1 |
| `Prewhitened(d)` | the score of detector d on AR-whitened, standardised values | that of d |
| e-CUSUM (`ECUSUM`) | log of the average of CUSUM e-detectors `max(0, C + λz − λ²/2)` over λ ∈ {0.25, 0.5, 1} on whitened innovations z | log 100 |

MeanShift is not from river: it is the simplest test of a shift in the mean error (with m = 1,
the test of Rombouts & Wilms), and with m > 1 it requires persistence (the degradation lasts m
windows in a row). At an equal false-alarm rate it is more powerful than the other detectors on a
shift in the mean (exp. 15), and MeanShift(3) ignores short-lived spikes (exp. 9).

Page-Hinkley and DDM reproduce `river` exactly: the first alarm falls on the same step on every
test stream (`tests/test_detectors.py`). ADWIN uses river's cut criterion on a fixed geometric
grid of split points instead of the exponential histogram; river checks for cuts only every
`clock = 32` steps, so the alarm differs from river's by a few dozen steps (in the tests, from 64
steps earlier to 32 steps later). All scores are vectorised over the first axis, so thousands of
bootstrap replicates are processed in one call.

**Whitening.** `ar_whiten` fits an AR model (Yule–Walker, order 2 by default) and the mean on
each reference, filters the whole series and divides by the reference innovation sd; under the
null the result is close to white noise with unit variance. `Prewhitened` runs any value detector
on it. During calibration the AR model is refitted on every bootstrap reference, so the p-values
stay valid. **e-CUSUM** works on the whitened innovations and is an e-detector in the sense of
Shin, Ramdas & Rinaldo: with exactly Gaussian white innovations, alarming at `score ≥ log A`
would give an average run length of at least A under the null. Real innovations are not exactly
Gaussian and the AR model is estimated, so its threshold is calibrated by bootstrap like any
other statistic. It also has an incremental form (`ECUSUM.start_stream`), so it can be checked at
every step.

Why not work with the binary signal directly: a black box with one threshold gives a p-value
with two values `{q, 1}`, where q is the probability of an alarm under the null. Online FDR
procedures over hundreds of streams need levels of 1e-4…1e-7, which such a p-value never reaches.

## 2. The test (`monitor.py`)

Time is split into windows of `window` steps (100 by default). Every stream has a reference
segment of `n_ref` steps (300) collected right after its last (re)training. At the end of each
window the detector is warmed up on the reference and run over the last `min(elapsed, horizon)`
windows (the horizon is 5 windows); the test statistic is the maximum score within the current
window. The horizon lets evidence accumulate, as in a detector running continuously in
production; without it online FDR finds almost nothing (see [the experiments](experiments.md#what-stage-1-showed)).
After an alarm the stream is retrained, a new reference is collected over the next `n_ref`
steps, and the stream is not monitored meanwhile.

**Sequential mode** (`StreamingMonitor(sequential=True)`, detectors with `start_stream`, i.e.
`ECUSUM`). The detector runs continuously from the end of the reference, and at every step its
current score is compared with the null distribution of the *largest score within a window*
(the same bootstrap as in the windowed mode). Alarming as soon as the p-value drops to α/K keeps
the probability of a false alarm in any window at α/K per model and α across the fleet — the
Bonferroni budget of the windowed mode — but without waiting for the window to end; the window
only sets the time unit of the budget. BH and online rules are not used in this mode.

**Streaming implementation.** When several models finish a window at the same step, their
window statistics are computed in one vectorised call per group of equal detectors;
`NullDistribution.pvalue_scalar` gives per-step p-values without numpy overhead.

The null hypothesis is defined by regimes: a test is null if the stream's mean is constant over
the whole span from the start of the reference to the end of the window. This is a synthetic
version of the "regime" definition of the null.

On real data the distribution always drifts, and the null "nothing changed" is almost always
false (exp. 8, 14). The main null is therefore **material degradation**: the error rose by no
more than a tolerance δ (`CalibrationConfig.tolerance`). During calibration the bootstrap
continuation of the series is shifted up by δ (for 0/1 errors, some zeros are turned into ones
so that the error rate rises by δ), so the p-value answers "did the error rise by more than δ".
The tolerance follows from the cost of a retrain: `tolerance_from_cost`. The ground truth on real
data is an oracle of future error: a test is null if the error over the next 1000 steps rose by
no more than δ (`error_rate_view`). References on which no test is possible (a constant series,
or an error rate + δ ≥ 1) get p = 1.

## 3. Calibration into a p-value (`calibration.py`, `bootstrap.py`)

The null distribution of the statistic is estimated from B bootstrap series (2000 by default)
resampled from the reference: both a pseudo-reference and pseudo-new data, so that their joint
variability matches the real test. The scores are causal, so one pass over a series of length
`n_ref + horizon·window` gives the null distributions for every horizon length at once: one
calibration per reference.

Resampling schemes: Künsch's moving-block bootstrap (`moving`), the Politis–Romano stationary
bootstrap (`stationary`), Bühlmann's AR-sieve bootstrap (`sieve`), the AR-sieve with parameter
uncertainty (`sieve_pu`, the default: the AR model is re-estimated on every replicate), and the
plain iid bootstrap as a control. Binary errors use the block bootstrap (detected from the data).
The block length follows the Politis–White formula for an AR(1) approximation; the AR order is
chosen by AIC.

The p-value is `(1 + #{T* ≥ T}) / (B + 1)`. Its minimum 1/(B+1) is too large for a correction
over hundreds of models, so when fewer than 10 bootstrap replicates reach the statistic, the
p-value is extrapolated from a tail fitted to the top 10% of the bootstrap sample (the approach
of Knijnenburg et al., 2009). By default the tail is a generalised Pareto distribution
(`tail="gpd"`): with it and B = 2000 the false-alarm rate of PH and KS holds within the error
bars, while the exponential tail understated it (exp. 6, 10). ADWIN's tail stays about three
times anti-conservative: its statistic is a maximum over many cuts.

## 4. Decision rules (`online_fdr.py`)

| Rule | What it controls |
|---|---|
| `uncorrected` | every stream at level α — current practice |
| `bonferroni` | α/K in every window: the probability of at least one false alarm in a window |
| `bh_window` | Benjamini–Hochberg within a window: FDR in every window |
| `storey_bh` | Storey's adaptive BH within a window: FDR in every window |
| `e_bh` | e-BH within a window (e = 0.5/√p): FDR under any dependence |
| `BatchBH` | BH within a window at levels that control FDR across all windows |
| `LOND` | online FDR, valid under positive dependence (PRDS) |
| `LORD++` | online FDR, the "wealth" is replenished at every discovery |
| `SAFFRON` | online FDR, adapts to the share of null hypotheses |
| `alpha-investing` | mFDR, Foster–Stine |
| `raw` | the river detector at its default threshold, no calibration |

Online rules receive the hypotheses of a window in a random order that does not depend on the
p-values.

Recommendation from the experiments: `bonferroni` holds its level under any correlation between
models (exp. 13); `bh_window` is faster when drifts come in clusters (exp. 4), but with correlated
models only together with `split_common` (exp. 18). Online FDR and e-BH are slower at an equal
number of false alarms (exp. 3, 12, 13).

### The fleet's common component (`preprocess.split_common`, `StreamingMonitor(split_common=True)`)

If the models' errors move together, a common spike shifts all p-values at once and produces a
burst of simultaneous false alarms. Each model is standardised by its own reference; the
cross-sectional median of the standardised values at each step is the common component, and a
residual is the model's value minus the median. The models' detectors watch the residuals, and
the common component is tested by its own calibrated detector in the same correction; its alarm
means an event across the whole fleet (`fleet_alarm`). The median assumes that fewer than half of
the models drift at once. On the real data sets the correlation of p-values between models is
0.4–0.9, and 0.17–0.33 after `split_common` (exp. 20).

## 5. Data and metrics (`streams.py`, `metrics.py`)

K streams are the error signals of K models: AR(1) in time with coefficient φ and a common factor
across streams with correlation ρ. In a share of the streams the mean shifts at a known time,
abruptly or gradually (in units of the standard deviation). A binary error signal for DDM is
built from the same latent process (base error rate 0.2).

Besides the mean shift there is a scenario with a feature, a label and a model
(`make_supervised_scenario`), where a shift in p(X), a shift in p(y|X), both at once and a cyclic
drift are set separately (exp. 11), and the fixed suite of 100 scenarios `benchmark_suite`
(exp. 21). The real data are fleets of models on INSECTS, Electricity, Airlines, Covertype and
hourly FX rates (`datasets.py`); `bucket_means` averages errors over time buckets (hour, day) so
that the daily cycle does not look like drift (exp. 16).

Metrics (`summarize`): FDR (the mean FDP over scenarios) and the FDP within a window, false alarms
per window, the probability of at least one false alarm in a window, MTFA, the share of missed
drifts (MDR), the mean delay (MTD), Bifet's MTR, event-level precision / recall / F1 as in the
benchmark of Cerqueira et al., and "steps on a stale model after a drift" — the delay with a miss
counted until the end of the stream. The last one is the direct cost of delay in practice.
