# Related work: where driftfdr stands in the literature

**English** · [Русский](related_work.ru.md)

A draft section for the paper. Numbers in square brackets are sources from the literature review
("Controlling false alarms in drift monitoring of systems with many ML models", in Russian;
the full list is in [References](#references) at the end),
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
- **e-detectors in driftfdr.** `ECUSUM` is a mixture CUSUM e-detector (Shin, Ramdas & Rinaldo)
  on AR-whitened innovations, and `StreamingMonitor(sequential=True)` checks it at every step
  with a Bonferroni-type rule across models, in the spirit of the e-d-Bonferroni procedure of
  [14]. Two differences: the threshold is calibrated by bootstrap to a false-alarm budget per
  window instead of being derived from the e-values (the innovations are not exactly Gaussian and
  the AR model is estimated), and the guarantee is the per-window probability of a false alarm
  rather than EOP. Experiment 23 compares it with the windowed detectors.
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

## References

Sources 1–25 are the reference list of the literature review, numbered as cited in the text.

1. Gama J., Žliobaitė I., Bifet A., Pechenizkiy M., Bouchachia A. A survey on concept drift adaptation // ACM Computing Surveys. 2014. Vol. 46, No. 4. P. 1–37.
2. Cerqueira V., Gomes H. M., Heyden M., Pfahringer B., Bifet A. A framework for evaluating and benchmarking concept drift detection methods // KDD ’26. 2026. arXiv:2606.07789.
3. Page E. S. Continuous inspection schemes // Biometrika. 1954. Vol. 41, No. 1/2. P. 100–115.
4. Gama J., Medas P., Castillo G., Rodrigues P. Learning with drift detection // Brazilian Symposium on Artificial Intelligence. Springer, 2004. P. 286–295.
5. Bifet A., Gavaldà R. Learning from time-changing data with adaptive windowing // Proceedings of the 2007 SIAM International Conference on Data Mining. 2007. P. 443–448.
6. dos Reis D. M., Flach P., Matwin S., Batista G. Fast unsupervised online drift detection using incremental Kolmogorov–Smirnov test // KDD ’16. 2016. P. 1545–1554.
7. Montiel J. et al. River: machine learning for streaming data in Python // Journal of Machine Learning Research. 2021. Vol. 22, No. 110. P. 1–8.
8. Bifet A. Classifier concept drift detection and the illusion of progress // International Conference on Artificial Intelligence and Soft Computing. Springer, 2017. P. 715–725.
9. Rombouts J., Wilms I. Monitoring machine learning forecasts for platform data streams. 2024. arXiv:2401.09144.
10. Hu Y. J., Rombouts J., Wilms I. MLOps monitoring at scale for digital platforms. 2025. arXiv:2504.16789.
11. Mei Y. Efficient scalable schemes for monitoring a large number of data streams // Biometrika. 2010. Vol. 97, No. 2. P. 419–433.
12. Chen J., Zhang W., Poor H. V. A false discovery rate oriented approach to parallel sequential change detection problems // IEEE Transactions on Signal Processing. 2020. Vol. 68. P. 1823–1836.
13. Chen Y., Li X. Compound sequential change-point detection in parallel data streams // Statistica Sinica. 2023. Vol. 33, No. 1. P. 453–474.
14. Dandapanthula S., Ramdas A. Multiple testing in multi-stream sequential change detection. 2025. arXiv:2501.04130.
15. Benjamini Y., Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing // Journal of the Royal Statistical Society: Series B. 1995. Vol. 57, No. 1. P. 289–300.
16. Foster D. P., Stine R. A. α-investing: a procedure for sequential control of expected false discoveries // Journal of the Royal Statistical Society: Series B. 2008. Vol. 70, No. 2. P. 429–444.
17. Javanmard A., Montanari A. Online rules for control of false discovery rate and false discovery exceedance // Annals of Statistics. 2018. Vol. 46, No. 2. P. 526–554.
18. Ramdas A., Yang F., Wainwright M. J., Jordan M. I. Online control of the false discovery rate with decaying memory // Advances in Neural Information Processing Systems (NeurIPS). 2017. Vol. 30.
19. Ramdas A., Zrnic T., Wainwright M. J., Jordan M. I. SAFFRON: an adaptive algorithm for online control of the false discovery rate // Proceedings of the 35th International Conference on Machine Learning (ICML). 2018.
20. Gang B., Sun W., Wang W. Structure-adaptive sequential testing for online false discovery rate control // Journal of the American Statistical Association. 2023. Vol. 118, No. 541. P. 732–745.
21. Rebjock Q., Kurt B., Januschowski T., Callot L. Online false discovery rate control for anomaly detection in time series // Advances in Neural Information Processing Systems (NeurIPS). 2021. Vol. 34.
22. Künsch H. R. The jackknife and the bootstrap for general stationary observations // Annals of Statistics. 1989. Vol. 17, No. 3. P. 1217–1241.
23. Politis D. N., Romano J. P. The stationary bootstrap // Journal of the American Statistical Association. 1994. Vol. 89, No. 428. P. 1303–1313.
24. Wu J., Apley D. W. Bootstrapped control limits for score-based concept drift control charts // Technometrics. 2026. arXiv:2507.16749.
25. Souza V. M. A., Reis D. M., Maletzke A. G., Batista G. E. A. P. A. Challenges in benchmarking stream learning algorithms with real-world data // Data Mining and Knowledge Discovery. 2020. Vol. 34. P. 1805–1858.

**Works outside the review** cited in this document and in the method description:

- Podkopaev A., Ramdas A. Tracking the risk of a deployed model and detecting harmful distribution
  shifts // ICLR. 2022.
- Shin J., Ramdas A., Rinaldo A. E-detectors: a nonparametric framework for sequential change
  detection. arXiv:2203.03532 (check the details of the published version).
- Wang R., Ramdas A. False discovery rate control with e-values // Journal of the Royal Statistical
  Society: Series B. 2022. Vol. 84, No. 3. P. 822–852.
- Storey J. D., Taylor J. E., Siegmund D. Strong control, conservative point estimation and
  simultaneous conservative consistency of false discovery rates: a unified approach // Journal of
  the Royal Statistical Society: Series B. 2004. Vol. 66, No. 1. P. 187–205.
- Zrnic T., Jiang D., Ramdas A., Jordan M. I. The power of batching in multiple hypothesis testing
  // AISTATS. 2020.
- Bühlmann P. Sieve bootstrap for time series // Bernoulli. 1997. Vol. 3, No. 2. P. 123–148.
- Politis D. N., White H. Automatic block-length selection for the dependent bootstrap //
  Econometric Reviews. 2004. Vol. 23, No. 1. P. 53–70.
- Knijnenburg T. A., Wessels L. F. A., Reinders M. J. T., Shmulevich I. Fewer permutations, more
  accurate P-values // Bioinformatics. 2009. Vol. 25, No. 12. P. i161–i168.
- NannyML: confidence-based performance estimation (CBPE), NannyML documentation,
  https://nannyml.readthedocs.io (there is no primary paper; the method is described in the documentation).
