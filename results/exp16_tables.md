## Airlines, почасовые ошибки, опорный отрезок неделя, окно сутки, δ = 0.05, 50 моделей

| detector | procedure | alarms | false_alarms | fdp | power_per_test |
|---|---|---|---|---|---|
| PH | uncorrected | 45 | 6.67 | 0.147 | 0.0573 |
| PH | bonferroni | 5.67 | 0 | 0 | 0.006 |
| PH | bh_window | 21.7 | 0 | 0 | 0.0264 |
| MeanShift(1) | uncorrected | 45 | 0 | 0 | 0.121 |
| MeanShift(1) | bonferroni | 19 | 0 | 0 | 0.0238 |
| MeanShift(1) | bh_window | 41.7 | 0 | 0 | 0.0992 |
| MeanShift(2) | uncorrected | 48.3 | 3 | 0.0618 | 0.0943 |
| MeanShift(2) | bonferroni | 30 | 0 | 0 | 0.0395 |
| MeanShift(2) | bh_window | 42.7 | 1.67 | 0.0395 | 0.0821 |
