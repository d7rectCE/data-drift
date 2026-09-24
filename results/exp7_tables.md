Метод калибровки для непрерывных сигналов: `sieve_pu`.

## A. Калибровка на первом стабильном концепте INSECTS

| detector | n_tests | far@0.05 | far@0.01 | far@0.001 | raw_far |
|---|---|---|---|---|---|
| DDM | 4.6e+03 | 0.00174 | 0 | 0 | 0.00507 |
| PH | 4.6e+03 | 0.000652 | 0 | 0 | 0.00594 |

## B. Правила принятия решений, α = 0.05, 50 моделей

| data | truth | detector | procedure | alarms | false_alarms | fdp | window_fdp | false_alarms_per_window | mdr | mean_delay | degraded_per_drift |
|---|---|---|---|---|---|---|---|---|---|---|---|
| elec2_events | injected | DDM | raw | 2.41e+03 | 2.4e+03 | 0.997 | 0.657 | 6.01 | 0.367 | 253 | 6.53e+03 |
| elec2_events | injected | DDM | uncorrected | 43.3 | 43.3 | 1 | 0.0467 | 0.108 | 1 | — | 2.01e+04 |
| elec2_events | injected | DDM | bonferroni | 3.67 | 3.67 | 1 | 0.00917 | 0.00917 | 1 | — | 2.01e+04 |
| elec2_events | injected | DDM | bh_window | 10 | 10 | 1 | 0.00333 | 0.025 | 1 | — | 2.01e+04 |
| elec2_events | injected | DDM | LORD++ | 0 | 0 | 0 | 0 | 0 | 1 | — | 2.01e+04 |
| elec2_events | injected | PH | raw | 3.28e+03 | 3.27e+03 | 0.997 | 0.683 | 8.17 | 0 | 273 | 273 |
| elec2_events | injected | PH | uncorrected | 379 | 373 | 0.984 | 0.158 | 0.932 | 0.4 | 2.55e+03 | 6.61e+03 |
| elec2_events | injected | PH | bonferroni | 201 | 195 | 0.97 | 0.0892 | 0.487 | 0.4 | 4.71e+03 | 8.26e+03 |
| elec2_events | injected | PH | bh_window | 258 | 254 | 0.985 | 0.051 | 0.636 | 0.6 | 1.86e+03 | 9.96e+03 |
| elec2_events | injected | PH | LORD++ | 191 | 186 | 0.973 | 0.0496 | 0.464 | 0.467 | 4.76e+03 | 9.03e+03 |
| elec2_sporadic | injected | DDM | raw | 2.4e+03 | 2.39e+03 | 0.998 | 0.672 | 5.98 | 0.433 | 332 | 7.87e+03 |
| elec2_sporadic | injected | DDM | uncorrected | 43.3 | 43 | 0.99 | 0.045 | 0.107 | 0.967 | 1.95e+04 | 1.94e+04 |
| elec2_sporadic | injected | DDM | bonferroni | 4 | 4 | 1 | 0.00917 | 0.01 | 1 | — | 1.94e+04 |
| elec2_sporadic | injected | DDM | bh_window | 6.67 | 6.67 | 1 | 0.005 | 0.0167 | 1 | — | 1.94e+04 |
| elec2_sporadic | injected | DDM | LORD++ | 0 | 0 | 0 | 0 | 0 | 1 | — | 1.94e+04 |
| elec2_sporadic | injected | PH | raw | 3.2e+03 | 3.19e+03 | 0.997 | 0.678 | 7.98 | 0 | 321 | 321 |
| elec2_sporadic | injected | PH | uncorrected | 370 | 365 | 0.986 | 0.149 | 0.912 | 0.467 | 1.78e+03 | 7.68e+03 |
| elec2_sporadic | injected | PH | bonferroni | 197 | 192 | 0.971 | 0.0897 | 0.479 | 0.433 | 1.95e+03 | 8.15e+03 |
| elec2_sporadic | injected | PH | bh_window | 248 | 244 | 0.984 | 0.0458 | 0.61 | 0.6 | 3.66e+03 | 1.09e+04 |
| elec2_sporadic | injected | PH | LORD++ | 168 | 163 | 0.971 | 0.0515 | 0.408 | 0.533 | 3.03e+03 | 9.77e+03 |
| insects | documented | DDM | raw | 176 | 76 | 0.431 | 0.022 | 0.154 | 0.6 | 7.93e+03 | 7.12e+03 |
| insects | documented | DDM | uncorrected | 365 | 161 | 0.44 | 0.085 | 0.325 | 0.131 | 5.22e+03 | 5.93e+03 |
| insects | documented | DDM | bonferroni | 88.3 | 55 | 0.626 | 0.00905 | 0.111 | 0.864 | 7.79e+03 | 7.56e+03 |
| insects | documented | DDM | bh_window | 313 | 121 | 0.388 | 0.0219 | 0.245 | 0.225 | 4.11e+03 | 5.85e+03 |
| insects | documented | DDM | LORD++ | 221 | 89.3 | 0.408 | 0.00978 | 0.18 | 0.472 | 6.38e+03 | 6.48e+03 |
| insects | documented | PH | raw | 911 | 772 | 0.847 | 0.228 | 1.56 | 0.305 | 4.64e+03 | 5.19e+03 |
| insects | documented | PH | uncorrected | 165 | 79.3 | 0.479 | 0.0308 | 0.16 | 0.653 | 2.24e+03 | 6.55e+03 |
| insects | documented | PH | bonferroni | 120 | 55.7 | 0.463 | 0.0104 | 0.112 | 0.741 | 1.28e+03 | 6.68e+03 |
| insects | documented | PH | bh_window | 123 | 54 | 0.439 | 0.00522 | 0.109 | 0.724 | 1.49e+03 | 6.67e+03 |
| insects | documented | PH | LORD++ | 117 | 51.3 | 0.438 | 0.00539 | 0.104 | 0.736 | 1.55e+03 | 6.73e+03 |
| insects | extended | DDM | raw | 176 | 8 | 0.0453 | 0.0128 | 0.0162 | 0.58 | 115 | 4.47e+03 |
| insects | extended | DDM | uncorrected | 365 | 84.7 | 0.231 | 0.0768 | 0.171 | 0.266 | 1.41e+03 | 3.73e+03 |
| insects | extended | DDM | bonferroni | 88.3 | 0.667 | 0.0072 | 0.000786 | 0.00135 | 0.779 | 1.15e+03 | 4.74e+03 |
| insects | extended | DDM | bh_window | 313 | 27.7 | 0.0878 | 0.0147 | 0.0559 | 0.282 | 530 | 3.69e+03 |
| insects | extended | DDM | LORD++ | 221 | 2.67 | 0.0118 | 0.00269 | 0.00539 | 0.453 | 773 | 4.1e+03 |
| insects | extended | PH | raw | 911 | 680 | 0.746 | 0.221 | 1.37 | 0.335 | 1.62e+03 | 3.26e+03 |
| insects | extended | PH | uncorrected | 165 | 25.7 | 0.154 | 0.0266 | 0.0519 | 0.649 | 162 | 4.11e+03 |
| insects | extended | PH | bonferroni | 120 | 4 | 0.0329 | 0.00572 | 0.00808 | 0.709 | 218 | 4.2e+03 |
| insects | extended | PH | bh_window | 123 | 2.67 | 0.0214 | 0.00303 | 0.00539 | 0.699 | 166 | 4.18e+03 |
| insects | extended | PH | LORD++ | 117 | 1.33 | 0.011 | 0.00135 | 0.00269 | 0.71 | 244 | 4.23e+03 |
