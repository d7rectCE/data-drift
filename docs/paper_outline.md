# Paper outline (draft)

**English** · [Русский](paper_outline.ru.md)

A skeleton for the paper: the claims, the evidence behind each (experiment numbers refer to
[experiments.md](experiments.md)) and what is still missing. Not a text yet.

## Working title

*Calibrated drift detection for fleets of ML models: p-values for practical detectors and when
multiplicity control pays off*

## Abstract (draft)

Drift detectors used in production (Page-Hinkley, DDM, ADWIN, KS) emit binary signals with an
unknown false-alarm rate, and across tens or hundreds of models their false alarms add up to
needless retrains. We turn such detectors into calibrated p-values by bootstrapping their internal
statistic on each model's reference segment with resampling schemes that respect autocorrelation,
and apply a multiplicity correction across the fleet. On synthetic fleets and five real data
sources the approach keeps the false-alarm rate near its nominal level where default thresholds
produce 26–97% false alarms. Multiplicity control within a window pays off in every benchmark
scenario, whereas online FDR procedures lose on delay; correlated models produce bursts of false
alarms that removing the fleet's common component eliminates; and a sequential e-detector with a
calibrated threshold detects abrupt shifts twice as fast as the calibrated classical detectors.

## Contributions and evidence

| # | claim | evidence |
|---|---|---|
| 1 | Practical detectors can be turned into calibrated p-values: the score is the least sensitivity at which the detector fires; an AR-sieve bootstrap with parameter uncertainty and a GPD tail hold the level for PH, KS and MeanShift; ADWIN is the exception | exp. 1, 6, 10, 15 |
| 2 | Compared at one actual false-alarm rate, detector rankings change; the simplest mean-shift test is the most powerful | exp. 1, 15 |
| 3 | Multiplicity control is the main lever (F1 0.13–0.33 → 0.86–0.88); Bonferroni or BH within a window suffices, online FDR (LORD++, SAFFRON, LOND) and e-BH lose on delay | exp. 2–4, 12, 13, 21 |
| 4 | Correlated models produce bursts of false alarms; the common component (`split_common`) removes them and is tested as a fleet-wide event; models on real data are strongly correlated (0.4–0.9), residuals only moderately (0.17–0.33), so a joint bootstrap is unnecessary | exp. 5, 18, 20, 21 |
| 5 | On real data the null must be "material degradation" (tolerance δ from the cost of a retrain), judged by a forward oracle; steps and windows must follow the data's cycle | exp. 7–9, 14, 16 |
| 6 | Against existing tools: Evidently's drift test alarms in 20–78% of non-degraded windows; NannyML's performance monitoring matches driftfdr's accuracy but not its control; CBPE fails without calibrated probabilities | exp. 17, 24 |
| 7 | A sequential e-CUSUM on whitened innovations with a bootstrap-calibrated threshold detects 26% faster at the same budget (2× on abrupt shifts); its look-back must be bounded or false alarms grow with time | exp. 23 |
| 8 | A 14-year fleet of FX volatility models: near-zero false alarms, material degradations caught, small ones indistinguishable from noise; river's default threshold is silent there and floods on 0/1 errors | exp. 22, 23 |

## Sections

1. **Introduction** — alarm fatigue across model fleets; the gap between practical detectors and
   multiple-testing theory (from the literature review).
2. **Related work** — [related_work.md](related_work.md).
3. **Method** — scores, calibration, the material null, decision rules, `split_common`, the
   sequential mode ([method.md](method.md)).
4. **Experimental setup** — synthetic fleets and `benchmark_suite`, real data, metrics.
5. **Results** — the claims above, one subsection each; figures: exp1 calibration, exp3 trade-off,
   exp5 dependence, exp7 INSECTS timeline, the demo.
6. **Limitations** — see the log.
7. **Conclusion and practical recommendations** — the configuration table of the README.

## Missing before submission

- A formal comparison of the per-window error with EOP (Dandapanthula–Ramdas); checking the works
  outside the review against the originals.
- Production logs of a heterogeneous fleet, or at least one more real multi-model data set whose
  models do not share a target.
- English figures (the figures in `results/` have Russian labels).
- Confidence intervals for the headline numbers where the experiments used few seeds.
