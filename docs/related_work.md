# Related work: where driftfdr stands in the literature

**English** · [Русский](related_work.ru.md)

A draft section for the paper. Numbers in square brackets are sources from the literature review
("Controlling false alarms in drift monitoring of systems with many ML models", in Russian),
which also contains a table matching the settings of the three lines of work. Here each line of
work is followed by what the driftfdr experiments showed (experiment numbers refer to the
[log](experiments.md)). Works outside the review are marked separately; their details must be
checked against the originals before the paper is submitted.

## Drift detectors in ML [1–8]

ADWIN, DDM, Page-Hinkley and KS emit a binary signal with an unknown actual false-alarm rate,
all the more so on an autocorrelated stream.

- **What driftfdr does.** Each detector is treated as a continuous statistic with one sensitivity
  parameter. Its null distribution is estimated by a bootstrap that accounts for
  autocorrelation, and the detector's signal becomes a p-value (exp. 1). For Page-Hinkley and DDM
  the statistic matches river.
- **What the experiments showed.** Under autocorrelation the plain bootstrap loses control, with
  several times more false alarms than promised; the block bootstrap is 1.5–3.5 times
  anti-conservative, and the AR-sieve comes closest to nominal (exp. 1). With parameter
  uncertainty and a GPD tail the level approximately holds for PH, KS and MeanShift, but not for
  ADWIN (exp. 10, 15). Feature detectors catch the wrong drift: they alarm on a harmless shift in
  p(X) and miss a harmful shift in p(y|X) (exp. 11).

## Detector benchmarks [2, 8]

Cerqueira et al. [2] compare detectors by F1 after tuning, but at different actual false-alarm
rates; the streams are shuffled, so there is no autocorrelation.

- **What driftfdr does.** All detectors are compared at the same actual false-alarm rate, on
  autocorrelated streams (exp. 15), and on a fixed suite of 100 scenarios (`benchmark_suite`,
  exp. 21). `summarize` reports the same event-level precision, recall and F1 as [2], so the
  numbers are comparable.
- **What the experiments showed.** Page-Hinkley, one of the worst in [2], is stronger than KS and
  DDM at an equal false-alarm rate and matches ADWIN by the third window; the most powerful is a
  simple mean-shift test (MeanShift) (exp. 15). This is not directly comparable with [2]
  (different streams and set of detectors), but it shows that the ranking of detectors depends on
  whether the false-alarm rate is aligned — the review's argument against comparing at different
  rates.
- **Default thresholds do not transfer between tasks.** river's Page-Hinkley at its defaults gives
  26–75% false retrains on 0/1 errors (exp. 14) and an F1 of 0.06 on the benchmark (exp. 21), yet
  on the daily loss of volatility models it does not alarm once in 14 years (exp. 22): the
  threshold λ = 50 is set in signal units. A calibrated threshold does not depend on the units.

## Monitoring many models in practice [9, 10]

Rombouts and Wilms test each of 32 districts at the 5% level without a correction, with an
actual test size of 6.7–7.4% and a 77% correlation of demand between districts.

- **What the experiments showed.** Without a correction false alarms grow linearly with the number
  of models; Bonferroni within a window holds its level under any correlation (exp. 2, 13). The
  review's expectation that false alarms under correlation come in bursts was confirmed: with BH,
  up to about fifteen in one window (exp. 18), dozens under strong correlation (exp. 5). On the
  real data sets the correlation of p-values between models is 0.4–0.9 (exp. 20). Splitting off
  the common component (`split_common`) removes most of it and the bursts with it; the common
  component is tested separately as a fleet-wide event (exp. 18, 20).

## Change detection in parallel streams [11–14]

The procedures of Mei, Chen–Zhang–Poor and Chen–Li build their own statistics from the streams'
observations and do not work with the signals of off-the-shelf detectors. Dandapanthula and
Ramdas [14] showed that with a finite ARL the worst case of classical FDR and FWER is trivial,
and introduced the error over patience (EOP).

- **Relation to driftfdr.** driftfdr does not replace these procedures; it connects river's
  detectors to the same logic. The error measure we end up recommending — the probability of at
  least one false alarm in a window (Bonferroni within a window, exp. 13) — is an error per unit
  of time rather than over the whole run, and in spirit is close to EOP: it does not degenerate as
  monitoring goes on. A formal comparison with EOP is an open question for the paper.
- The share of false alarms over the whole run (`fdp`) is closer to the setting of
  Chen–Zhang–Poor [12]. BH within a window does not control it with correlated models (exp. 13).

## Online multiple testing [15–21]

Alpha-investing, LORD and SAFFRON need valid p-values; Rebjock et al. [21] applied online FDR to
anomaly scores in time series.

- **What the experiments showed.** When the statistic accumulates evidence over several windows,
  online procedures (LORD++, SAFFRON, LOND) lose on delay at an equal number of false alarms,
  because they spend the error budget unevenly over time (exp. 3, 12). BH within a window pays
  off when drifts come in clusters (exp. 4); e-BH, valid under any dependence, is too cautious:
  a third of the drifts are missed (exp. 13).
- **Difference from [21].** We calibrate the statistics of drift detectors on the model's error
  rather than anomaly scores, and we treat shifts in p(X) and p(y|X) separately (exp. 11).

## Calibration under dependence [22–24]

The block [22] and stationary [23] bootstraps preserve dependence; Wu and Apley [24] showed that
the nested bootstrap underestimates the variability of the statistic and proposed a correction
for a single MEWMA chart.

- **What driftfdr does.** Block, stationary and AR-sieve bootstraps for a family of detectors;
  the uncertainty of the parameter estimates is accounted for by re-estimating the AR model on
  every replicate (`sieve_pu`) — the same problem as in [24], solved differently; the 0.632
  correction of [24] is not implemented. The deep tail needed for a correction over thousands of
  models is extrapolated with a GPD (exp. 6, 10).

## Outside the review (check against the originals)

- **Podkopaev and Ramdas, "Tracking the risk of a deployed model and detecting harmful
  distribution shifts" (ICLR 2022).** A sequential test that the model's risk rose by more than a
  given tolerance, valid at any time. It is the same idea as our null of "material degradation"
  with a tolerance δ (exp. 8): alarm on harmful shifts, not on any shift. The differences: they
  have one model and their own sequential test; we calibrate existing detectors, for a fleet of
  models, with a multiplicity correction. The paper should cite it as the closest work on the
  formulation of the null.
- **CBPE in NannyML (label-free performance estimation from predicted probabilities).** Estimates
  a model's accuracy before labels arrive, assuming calibrated probabilities and no shift in
  p(y|X). Not a competitor but a possible source of signal: with delayed labels driftfdr can
  monitor the CBPE estimate instead of the actual error. By construction CBPE cannot see a shift
  in p(y|X) — exactly the case that proved harmful in exp. 11.
- **NannyML's performance monitoring (with labels)** is compared directly in exp. 17: at a
  comparable false-alarm rate it catches degradations about as well; its ±3σ threshold is fixed
  and does not account for the number of models.
