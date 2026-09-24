## Четыре реальных набора, истина — существенное ухудшение (δ = 0.05), 50 моделей

`raw` — Page-Hinkley river с порогом по умолчанию на ряде ошибок 0/1, без калибровки.

| data | detector | rule | alarms_per_model_10k | fdp | power_per_test |
|---|---|---|---|---|---|
| INSECTS | PH | raw | 0.401 | 0.5 | 0.101 |
| INSECTS | PH | uncorrected | 0.203 | 0.0098 | 0.101 |
| INSECTS | PH | bonferroni | 0.201 | 0 | 0.0998 |
| INSECTS | PH | bh_window | 0.201 | 0 | 0.101 |
| INSECTS | MeanShift(3) | uncorrected | 0.203 | 0.0098 | 0.0838 |
| INSECTS | MeanShift(3) | bonferroni | 0.201 | 0 | 0.0837 |
| INSECTS | MeanShift(3) | bh_window | 0.201 | 0 | 0.0837 |
| Electricity | PH | raw | 1.82 | 0.562 | 0.132 |
| Electricity | PH | uncorrected | 0.789 | 0.335 | 0.0584 |
| Electricity | PH | bonferroni | 0.414 | 0.295 | 0.0099 |
| Electricity | PH | bh_window | 0.494 | 0.272 | 0.0248 |
| Electricity | MeanShift(3) | uncorrected | 0.429 | 0.0756 | 0.057 |
| Electricity | MeanShift(3) | bonferroni | 0.251 | 0.267 | 0.00915 |
| Electricity | MeanShift(3) | bh_window | 0.322 | 0 | 0.0259 |
| Airlines | PH | raw | 0.168 | 0.263 | 0.015 |
| Airlines | PH | uncorrected | 0.235 | 0.366 | 0.0333 |
| Airlines | PH | bonferroni | 0.0516 | 0.332 | 0.00216 |
| Airlines | PH | bh_window | 0.178 | 0.31 | 0.0139 |
| Airlines | MeanShift(3) | uncorrected | 0.246 | 0.353 | 0.0423 |
| Airlines | MeanShift(3) | bonferroni | 0.0547 | 0.348 | 0.00249 |
| Airlines | MeanShift(3) | bh_window | 0.179 | 0.319 | 0.0174 |
| Covertype | PH | raw | 1.34 | 0.751 | 0.128 |
| Covertype | PH | uncorrected | 0.478 | 0.503 | 0.0464 |
| Covertype | PH | bonferroni | 0.183 | 0.293 | 0.0188 |
| Covertype | PH | bh_window | 0.204 | 0.298 | 0.0237 |
| Covertype | MeanShift(3) | uncorrected | 0.233 | 0.235 | 0.123 |
| Covertype | MeanShift(3) | bonferroni | 0.176 | 0.15 | 0.054 |
| Covertype | MeanShift(3) | bh_window | 0.195 | 0.209 | 0.0738 |
