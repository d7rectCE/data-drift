## Лучшая задержка в пределах бюджета ложных переобучений

`vs_fixed_threshold` — отношение к лучшему порогу на поток без поправки при том же бюджете.

| n_streams | scenario | procedure | budget | degraded_per_drift | alpha | vs_fixed_threshold |
|---|---|---|---|---|---|---|
| 100 | 1 event | BatchBH | 0.001 | 373 | 0.2 | 0.732 |
| 100 | 1 event | BatchBH | 0.01 | 373 | 0.2 | 1.73 |
| 100 | 1 event | LORD++ | 0.001 | 312 | 0.05 | 0.612 |
| 100 | 1 event | LORD++ | 0.01 | 262 | 0.2 | 1.22 |
| 100 | 1 event | bh_window | 0.001 | 401 | 0.001 | 0.787 |
| 100 | 1 event | bh_window | 0.01 | 168 | 0.05 | 0.779 |
| 100 | 1 event | bonferroni | 0.001 | 504 | 0.001 | 0.99 |
| 100 | 1 event | bonferroni | 0.01 | 213 | 0.2 | 0.988 |
| 100 | 1 event | storey_bh | 0.001 | 612 | 0.0001 | 1.2 |
| 100 | 1 event | storey_bh | 0.01 | 236 | 0.01 | 1.1 |
| 100 | 1 event | uncorrected | 0.001 | 509 | 1e-05 | 1 |
| 100 | 1 event | uncorrected | 0.01 | 215 | 0.001 | 1 |
| 100 | 2 events | BatchBH | 0.001 | 295 | 0.2 | 0.725 |
| 100 | 2 events | BatchBH | 0.01 | 295 | 0.2 | 1.31 |
| 100 | 2 events | LORD++ | 0.001 | 362 | 0.01 | 0.889 |
| 100 | 2 events | LORD++ | 0.01 | 229 | 0.2 | 1.02 |
| 100 | 2 events | bh_window | 0.001 | 337 | 0.001 | 0.83 |
| 100 | 2 events | bh_window | 0.01 | 183 | 0.05 | 0.815 |
| 100 | 2 events | bonferroni | 0.001 | 402 | 0.001 | 0.988 |
| 100 | 2 events | bonferroni | 0.01 | 209 | 0.2 | 0.93 |
| 100 | 2 events | storey_bh | 0.001 | 296 | 0.001 | 0.729 |
| 100 | 2 events | storey_bh | 0.01 | 242 | 0.01 | 1.07 |
| 100 | 2 events | uncorrected | 0.001 | 406 | 1e-05 | 1 |
| 100 | 2 events | uncorrected | 0.01 | 225 | 0.001 | 1 |
| 100 | sporadic | BatchBH | 0.001 | 351 | 0.2 | 1.32 |
| 100 | sporadic | BatchBH | 0.01 | 351 | 0.2 | 1.72 |
| 100 | sporadic | LORD++ | 0.001 | 429 | 0.01 | 1.61 |
| 100 | sporadic | LORD++ | 0.01 | 258 | 0.2 | 1.27 |
| 100 | sporadic | bh_window | 0.001 | 414 | 0.001 | 1.56 |
| 100 | sporadic | bh_window | 0.01 | 213 | 0.05 | 1.05 |
| 100 | sporadic | bonferroni | 0.001 | 266 | 0.01 | 1 |
| 100 | sporadic | bonferroni | 0.01 | 203 | 0.1 | 1 |
| 100 | sporadic | storey_bh | 0.001 | 389 | 0.001 | 1.46 |
| 100 | sporadic | storey_bh | 0.01 | 250 | 0.01 | 1.23 |
| 100 | sporadic | uncorrected | 0.001 | 266 | 0.0001 | 1 |
| 100 | sporadic | uncorrected | 0.01 | 203 | 0.001 | 1 |
| 500 | 1 event | BatchBH | 0.001 | 356 | 0.2 | 1.31 |
| 500 | 1 event | BatchBH | 0.01 | 356 | 0.2 | 1.72 |
| 500 | 1 event | LORD++ | 0.001 | 254 | 0.1 | 0.937 |
| 500 | 1 event | LORD++ | 0.01 | 251 | 0.2 | 1.21 |
| 500 | 1 event | bh_window | 0.001 | 216 | 0.01 | 0.798 |
| 500 | 1 event | bh_window | 0.01 | 172 | 0.05 | 0.831 |
| 500 | 1 event | bonferroni | 0.001 | 246 | 0.1 | 0.907 |
| 500 | 1 event | bonferroni | 0.01 | 227 | 0.2 | 1.1 |
| 500 | 1 event | storey_bh | 0.001 | 356 | 0.001 | 1.31 |
| 500 | 1 event | storey_bh | 0.01 | 356 | 0.001 | 1.72 |
| 500 | 1 event | uncorrected | 0.001 | 271 | 0.0001 | 1 |
| 500 | 1 event | uncorrected | 0.01 | 206 | 0.001 | 1 |
| 500 | 2 events | BatchBH | 0.001 | 342 | 0.2 | 1.23 |
| 500 | 2 events | BatchBH | 0.01 | 342 | 0.2 | 1.66 |
| 500 | 2 events | LORD++ | 0.001 | 274 | 0.05 | 0.981 |
| 500 | 2 events | LORD++ | 0.01 | 244 | 0.2 | 1.18 |
| 500 | 2 events | bh_window | 0.001 | 235 | 0.01 | 0.84 |
| 500 | 2 events | bh_window | 0.01 | 194 | 0.05 | 0.944 |
| 500 | 2 events | bonferroni | 0.001 | 277 | 0.05 | 0.99 |
| 500 | 2 events | bonferroni | 0.01 | 231 | 0.2 | 1.12 |
| 500 | 2 events | storey_bh | 0.001 | 389 | 0.001 | 1.39 |
| 500 | 2 events | storey_bh | 0.01 | 389 | 0.001 | 1.89 |
| 500 | 2 events | uncorrected | 0.001 | 279 | 0.0001 | 1 |
| 500 | 2 events | uncorrected | 0.01 | 206 | 0.001 | 1 |
| 500 | sporadic | BatchBH | 0.001 | 375 | 0.1 | 1.41 |
| 500 | sporadic | BatchBH | 0.01 | 318 | 0.2 | 1.6 |
| 500 | sporadic | LORD++ | 0.001 | 347 | 0.01 | 1.31 |
| 500 | sporadic | LORD++ | 0.01 | 229 | 0.2 | 1.15 |
| 500 | sporadic | bh_window | 0.001 | 266 | 0.01 | 1 |
| 500 | sporadic | bh_window | 0.01 | 219 | 0.05 | 1.1 |
| 500 | sporadic | bonferroni | 0.001 | 243 | 0.1 | 0.916 |
| 500 | sporadic | bonferroni | 0.01 | 217 | 0.2 | 1.09 |
| 500 | sporadic | storey_bh | 0.001 | 376 | 0.001 | 1.42 |
| 500 | sporadic | storey_bh | 0.01 | 271 | 0.01 | 1.37 |
| 500 | sporadic | uncorrected | 0.001 | 265 | 0.0001 | 1 |
| 500 | sporadic | uncorrected | 0.01 | 199 | 0.001 | 1 |

## Все правила при α = 0.05

| n_streams | scenario | procedure | false_alarms_per_1k_stream_steps | degraded_per_drift | fdr | window_fdp | mdr | mean_delay |
|---|---|---|---|---|---|---|---|---|
| 100 | 1 event | BatchBH | 0.000355 | 469 | 0.00794 | 0.00355 | 0.0417 | 379 |
| 100 | 1 event | LORD++ | 0 | 312 | 0 | 0 | 0.00833 | 291 |
| 100 | 1 event | bh_window | 0.00851 | 168 | 0.137 | 0.0363 | 0 | 168 |
| 100 | 1 event | bonferroni | 0.00532 | 235 | 0.0972 | 0.0496 | 0 | 235 |
| 100 | 1 event | storey_bh | 0.0699 | 175 | 0.395 | 0.0297 | 0 | 175 |
| 100 | 1 event | uncorrected | 0.133 | 130 | 0.729 | 0.514 | 0 | 130 |
| 100 | 2 events | BatchBH | 0.000355 | 407 | 0.00794 | 0.00355 | 0.025 | 372 |
| 100 | 2 events | LORD++ | 0.00106 | 268 | 0.0217 | 0.00217 | 0 | 268 |
| 100 | 2 events | bh_window | 0.00851 | 183 | 0.135 | 0.0243 | 0 | 183 |
| 100 | 2 events | bonferroni | 0.00461 | 216 | 0.0827 | 0.0368 | 0 | 216 |
| 100 | 2 events | storey_bh | 0.0411 | 172 | 0.349 | 0.0372 | 0 | 172 |
| 100 | 2 events | uncorrected | 0.128 | 136 | 0.713 | 0.468 | 0 | 136 |
| 100 | sporadic | BatchBH | 0 | 503 | 0 | 0 | 0.05 | 391 |
| 100 | sporadic | LORD++ | 0.00142 | 312 | 0.031 | 0.00827 | 0 | 312 |
| 100 | sporadic | bh_window | 0.00674 | 213 | 0.102 | 0.0254 | 0 | 213 |
| 100 | sporadic | bonferroni | 0.00213 | 219 | 0.0413 | 0.0142 | 0 | 219 |
| 100 | sporadic | storey_bh | 0.0337 | 211 | 0.303 | 0.0431 | 0 | 211 |
| 100 | sporadic | uncorrected | 0.137 | 127 | 0.739 | 0.477 | 0 | 127 |
| 500 | 1 event | BatchBH | 0 | 471 | 0 | 0 | 0.0367 | 398 |
| 500 | 1 event | LORD++ | 0.000355 | 272 | 0.00815 | 0.000368 | 0 | 272 |
| 500 | 1 event | bh_window | 0.00603 | 172 | 0.0862 | 0.0222 | 0 | 172 |
| 500 | 1 event | bonferroni | 0.000355 | 270 | 0.00806 | 0.0177 | 0 | 270 |
| 500 | 1 event | storey_bh | 0.0365 | 171 | 0.277 | 0.0325 | 0 | 171 |
| 500 | 1 event | uncorrected | 0.119 | 145 | 0.657 | 0.726 | 0.00333 | 137 |
| 500 | 2 events | BatchBH | 0 | 461 | 0 | 0 | 0.0383 | 403 |
| 500 | 2 events | LORD++ | 0.000567 | 274 | 0.0127 | 0.000462 | 0.00167 | 270 |
| 500 | 2 events | bh_window | 0.00504 | 194 | 0.0792 | 0.0218 | 0.00333 | 188 |
| 500 | 2 events | bonferroni | 0.000355 | 277 | 0.00806 | 0.0177 | 0.005 | 266 |
| 500 | 2 events | storey_bh | 0.0415 | 201 | 0.318 | 0.0305 | 0.00167 | 197 |
| 500 | 2 events | uncorrected | 0.121 | 134 | 0.673 | 0.726 | 0.00167 | 131 |
| 500 | sporadic | BatchBH | 7.09e-05 | 417 | 0.00165 | 0.000443 | 0.0283 | 372 |
| 500 | sporadic | LORD++ | 0.00135 | 271 | 0.0275 | 0.00501 | 0.00167 | 267 |
| 500 | sporadic | bh_window | 0.00475 | 219 | 0.0842 | 0.0318 | 0 | 219 |
| 500 | sporadic | bonferroni | 0.000496 | 264 | 0.0111 | 0.0193 | 0.00167 | 260 |
| 500 | sporadic | storey_bh | 0.0686 | 238 | 0.355 | 0.0306 | 0.00833 | 223 |
| 500 | sporadic | uncorrected | 0.137 | 127 | 0.719 | 0.618 | 0.00167 | 124 |
