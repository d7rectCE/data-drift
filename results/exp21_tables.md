## Бенчмарк: 25 случаев × 4 сидов, α = 0.05, F1 при задержке ≤ 1000 шагов

### Все сценарии, по убыванию F1

| detector | view | rule | false_alarms_per_1k_stream_steps | p_any_false_alarm_per_window | fdp | precision | recall | f1 | mean_delay | degraded_per_drift |
|---|---|---|---|---|---|---|---|---|---|---|
| MeanShift(3) | остатки | bh_window | 0.00164 | 0.0155 | 0.0981 | 0.854 | 0.895 | 0.884 | 507 | 580 |
| PH | остатки | bh_window | 0.00147 | 0.0134 | 0.0889 | 0.853 | 0.886 | 0.877 | 460 | 551 |
| MeanShift(3) | остатки | bonferroni | 0.00121 | 0.0121 | 0.0822 | 0.849 | 0.866 | 0.867 | 553 | 635 |
| PH | остатки | bonferroni | 0.00106 | 0.0106 | 0.0743 | 0.854 | 0.856 | 0.862 | 499 | 612 |
| KS | остатки | bonferroni | 0.00319 | 0.0313 | 0.152 | 0.785 | 0.856 | 0.823 | 554 | 677 |
| KS | остатки | bh_window | 0.00502 | 0.0423 | 0.214 | 0.737 | 0.877 | 0.807 | 527 | 624 |
| MeanShift(1) | остатки | bonferroni | 0.00338 | 0.0334 | 0.179 | 0.742 | 0.818 | 0.787 | 453 | 612 |
| MeanShift(1) | остатки | bh_window | 0.00447 | 0.0396 | 0.209 | 0.72 | 0.84 | 0.785 | 416 | 561 |
| PH | сырые | bonferroni | 0.00149 | 0.013 | 0.0576 | 0.82 | 0.77 | 0.775 | 575 | 773 |
| PH | сырые | bh_window | 0.00751 | 0.0183 | 0.134 | 0.769 | 0.832 | 0.77 | 534 | 658 |
| MeanShift(3) | сырые | bonferroni | 0.00194 | 0.0166 | 0.0872 | 0.773 | 0.769 | 0.762 | 629 | 772 |
| KS | сырые | bonferroni | 0.0023 | 0.0191 | 0.0944 | 0.773 | 0.766 | 0.761 | 644 | 806 |
| MeanShift(3) | сырые | bh_window | 0.00713 | 0.0198 | 0.141 | 0.739 | 0.812 | 0.754 | 586 | 671 |
| KS | сырые | bh_window | 0.0084 | 0.0245 | 0.172 | 0.719 | 0.803 | 0.745 | 615 | 723 |
| MeanShift(1) | сырые | bonferroni | 0.00421 | 0.033 | 0.163 | 0.705 | 0.743 | 0.709 | 556 | 757 |
| ADWIN | остатки | bonferroni | 0.0146 | 0.137 | 0.435 | 0.527 | 0.885 | 0.673 | 398 | 506 |
| ADWIN | сырые | bonferroni | 0.0125 | 0.096 | 0.381 | 0.548 | 0.805 | 0.654 | 496 | 627 |
| MeanShift(1) | сырые | bh_window | 0.0239 | 0.043 | 0.317 | 0.579 | 0.815 | 0.641 | 518 | 619 |
| ADWIN | остатки | bh_window | 0.0188 | 0.149 | 0.49 | 0.479 | 0.909 | 0.639 | 375 | 457 |
| ADWIN | сырые | LORD++ | 0.00236 | 0.0149 | 0.0959 | 0.776 | 0.646 | 0.635 | 576 | 991 |
| ADWIN | сырые | bh_window | 0.0283 | 0.0981 | 0.47 | 0.472 | 0.831 | 0.591 | 474 | 575 |
| KS | сырые | LORD++ | 0.000149 | 0.00149 | 0.00697 | 0.76 | 0.517 | 0.522 | 825 | 1.3e+03 |
| PH | сырые | LORD++ | 0.000106 | 0.00106 | 0.00475 | 0.843 | 0.472 | 0.479 | 656 | 1.35e+03 |
| MeanShift(3) | сырые | LORD++ | 0.000128 | 0.00106 | 0.0053 | 0.873 | 0.471 | 0.474 | 642 | 1.37e+03 |
| MeanShift(1) | сырые | LORD++ | 0.00017 | 0.00149 | 0.00685 | 0.771 | 0.352 | 0.356 | 686 | 1.53e+03 |
| PH | сырые | uncorrected | 0.1 | 0.42 | 0.758 | 0.202 | 0.986 | 0.333 | 310 | 319 |
| MeanShift(3) | сырые | uncorrected | 0.14 | 0.513 | 0.824 | 0.153 | 0.986 | 0.262 | 371 | 387 |
| KS | сырые | uncorrected | 0.135 | 0.601 | 0.845 | 0.142 | 0.986 | 0.252 | 380 | 385 |
| MeanShift(1) | сырые | uncorrected | 0.263 | 0.644 | 0.897 | 0.0795 | 0.99 | 0.149 | 251 | 257 |
| ADWIN | сырые | uncorrected | 0.276 | 0.806 | 0.915 | 0.07 | 0.972 | 0.135 | 290 | 316 |
| PH river по умолчанию | сырые | порог | 0.734 | 0.891 | 0.96 | 0.0275 | 0.998 | 0.0557 | 208 | 211 |

### Сценарий без дрейфа: ложные тревоги

| detector | view | rule | false_alarms_per_1k_stream_steps | p_any_false_alarm_per_window |
|---|---|---|---|---|
| PH | сырые | uncorrected | 0.0809 | 0.415 |
| PH | сырые | bonferroni | 0 | 0 |
| PH | сырые | bh_window | 0 | 0 |
| PH | сырые | LORD++ | 0 | 0 |
| PH | остатки | bonferroni | 0.0016 | 0.016 |
| PH | остатки | bh_window | 0.0016 | 0.016 |
| KS | сырые | uncorrected | 0.12 | 0.59 |
| KS | сырые | bonferroni | 0 | 0 |
| KS | сырые | bh_window | 0 | 0 |
| KS | сырые | LORD++ | 0 | 0 |
| KS | остатки | bonferroni | 0.00372 | 0.0372 |
| KS | остатки | bh_window | 0.00372 | 0.0372 |
| ADWIN | сырые | uncorrected | 0.265 | 0.878 |
| ADWIN | сырые | bonferroni | 0.0122 | 0.0957 |
| ADWIN | сырые | bh_window | 0.0176 | 0.0798 |
| ADWIN | сырые | LORD++ | 0 | 0 |
| ADWIN | остатки | bonferroni | 0.0112 | 0.101 |
| ADWIN | остатки | bh_window | 0.0128 | 0.106 |
| MeanShift(1) | сырые | uncorrected | 0.232 | 0.665 |
| MeanShift(1) | сырые | bonferroni | 0.000532 | 0.00532 |
| MeanShift(1) | сырые | bh_window | 0.000532 | 0.00532 |
| MeanShift(1) | сырые | LORD++ | 0 | 0 |
| MeanShift(1) | остатки | bonferroni | 0.00372 | 0.0372 |
| MeanShift(1) | остатки | bh_window | 0.00585 | 0.0426 |
| MeanShift(3) | сырые | uncorrected | 0.103 | 0.452 |
| MeanShift(3) | сырые | bonferroni | 0.000532 | 0.00532 |
| MeanShift(3) | сырые | bh_window | 0.000532 | 0.00532 |
| MeanShift(3) | сырые | LORD++ | 0 | 0 |
| MeanShift(3) | остатки | bonferroni | 0.0016 | 0.016 |
| MeanShift(3) | остатки | bh_window | 0.0016 | 0.016 |
| PH river по умолчанию | сырые | порог | 0.706 | 0.91 |

### F1 по фактору `drift_type`

| detector | view | rule | drift_type = abrupt | drift_type = gradual |
|---|---|---|---|---|
| MeanShift(3) | остатки | bh_window | 0.918 | 0.849 |
| PH | остатки | bh_window | 0.911 | 0.843 |
| MeanShift(3) | остатки | bonferroni | 0.9 | 0.833 |
| PH | остатки | bonferroni | 0.894 | 0.83 |
| KS | остатки | bonferroni | 0.859 | 0.786 |
| KS | остатки | bh_window | 0.831 | 0.782 |
| MeanShift(1) | остатки | bonferroni | 0.823 | 0.751 |
| MeanShift(1) | остатки | bh_window | 0.815 | 0.756 |
| PH | сырые | bonferroni | 0.822 | 0.728 |
| PH | сырые | bh_window | 0.816 | 0.725 |
| MeanShift(3) | сырые | bonferroni | 0.815 | 0.709 |
| KS | сырые | bonferroni | 0.814 | 0.708 |
| MeanShift(3) | сырые | bh_window | 0.799 | 0.709 |
| KS | сырые | bh_window | 0.797 | 0.693 |
| MeanShift(1) | сырые | bonferroni | 0.749 | 0.668 |
| ADWIN | остатки | bonferroni | 0.715 | 0.63 |
| ADWIN | сырые | bonferroni | 0.709 | 0.599 |
| MeanShift(1) | сырые | bh_window | 0.675 | 0.608 |
| ADWIN | остатки | bh_window | 0.681 | 0.597 |
| ADWIN | сырые | LORD++ | 0.691 | 0.578 |
| ADWIN | сырые | bh_window | 0.624 | 0.558 |
| KS | сырые | LORD++ | 0.552 | 0.491 |
| PH | сырые | LORD++ | 0.521 | 0.436 |
| MeanShift(3) | сырые | LORD++ | 0.498 | 0.449 |
| MeanShift(1) | сырые | LORD++ | 0.408 | 0.304 |
| PH | сырые | uncorrected | 0.361 | 0.304 |
| MeanShift(3) | сырые | uncorrected | 0.272 | 0.252 |
| KS | сырые | uncorrected | 0.251 | 0.254 |
| MeanShift(1) | сырые | uncorrected | 0.147 | 0.151 |
| ADWIN | сырые | uncorrected | 0.139 | 0.131 |
| PH river по умолчанию | сырые | порог | 0.0563 | 0.055 |

### F1 по фактору `magnitude`

| detector | view | rule | magnitude = 0.5 | magnitude = 1.0 |
|---|---|---|---|---|
| MeanShift(3) | остатки | bh_window | 0.8 | 0.967 |
| PH | остатки | bh_window | 0.796 | 0.959 |
| MeanShift(3) | остатки | bonferroni | 0.757 | 0.976 |
| PH | остатки | bonferroni | 0.755 | 0.968 |
| KS | остатки | bonferroni | 0.73 | 0.915 |
| KS | остатки | bh_window | 0.732 | 0.881 |
| MeanShift(1) | остатки | bonferroni | 0.652 | 0.922 |
| MeanShift(1) | остатки | bh_window | 0.67 | 0.901 |
| PH | сырые | bonferroni | 0.589 | 0.961 |
| PH | сырые | bh_window | 0.646 | 0.895 |
| MeanShift(3) | сырые | bonferroni | 0.564 | 0.959 |
| KS | сырые | bonferroni | 0.566 | 0.956 |
| MeanShift(3) | сырые | bh_window | 0.595 | 0.914 |
| KS | сырые | bh_window | 0.585 | 0.905 |
| MeanShift(1) | сырые | bonferroni | 0.505 | 0.912 |
| ADWIN | остатки | bonferroni | 0.62 | 0.725 |
| ADWIN | сырые | bonferroni | 0.515 | 0.793 |
| MeanShift(1) | сырые | bh_window | 0.502 | 0.781 |
| ADWIN | остатки | bh_window | 0.605 | 0.674 |
| ADWIN | сырые | LORD++ | 0.326 | 0.944 |
| ADWIN | сырые | bh_window | 0.475 | 0.706 |
| KS | сырые | LORD++ | 0.0788 | 0.965 |
| PH | сырые | LORD++ | 0.0469 | 0.91 |
| MeanShift(3) | сырые | LORD++ | 0.0336 | 0.914 |
| MeanShift(1) | сырые | LORD++ | 0.00657 | 0.705 |
| PH | сырые | uncorrected | 0.336 | 0.33 |
| MeanShift(3) | сырые | uncorrected | 0.28 | 0.244 |
| KS | сырые | uncorrected | 0.239 | 0.266 |
| MeanShift(1) | сырые | uncorrected | 0.156 | 0.143 |
| ADWIN | сырые | uncorrected | 0.132 | 0.138 |
| PH river по умолчанию | сырые | порог | 0.057 | 0.0543 |

### F1 по фактору `rho`

| detector | view | rule | rho = 0.0 | rho = 0.3 | rho = 0.6 |
|---|---|---|---|---|---|
| MeanShift(3) | остатки | bh_window | 0.812 | 0.89 | 0.949 |
| PH | остатки | bh_window | 0.825 | 0.861 | 0.946 |
| MeanShift(3) | остатки | bonferroni | 0.789 | 0.86 | 0.951 |
| PH | остатки | bonferroni | 0.78 | 0.852 | 0.952 |
| KS | остатки | bonferroni | 0.751 | 0.812 | 0.905 |
| KS | остатки | bh_window | 0.747 | 0.788 | 0.885 |
| MeanShift(1) | остатки | bonferroni | 0.704 | 0.766 | 0.891 |
| MeanShift(1) | остатки | bh_window | 0.72 | 0.762 | 0.873 |
| PH | сырые | bonferroni | 0.802 | 0.785 | 0.739 |
| PH | сырые | bh_window | 0.822 | 0.772 | 0.717 |
| MeanShift(3) | сырые | bonferroni | 0.802 | 0.759 | 0.724 |
| KS | сырые | bonferroni | 0.776 | 0.779 | 0.728 |
| MeanShift(3) | сырые | bh_window | 0.818 | 0.751 | 0.694 |
| KS | сырые | bh_window | 0.801 | 0.758 | 0.676 |
| MeanShift(1) | сырые | bonferroni | 0.714 | 0.709 | 0.703 |
| ADWIN | остатки | bonferroni | 0.629 | 0.66 | 0.729 |
| ADWIN | сырые | bonferroni | 0.663 | 0.653 | 0.647 |
| MeanShift(1) | сырые | bh_window | 0.73 | 0.607 | 0.587 |
| ADWIN | остатки | bh_window | 0.611 | 0.62 | 0.686 |
| ADWIN | сырые | LORD++ | 0.663 | 0.631 | 0.611 |
| ADWIN | сырые | bh_window | 0.656 | 0.609 | 0.507 |
| KS | сырые | LORD++ | 0.518 | 0.52 | 0.527 |
| PH | сырые | LORD++ | 0.488 | 0.501 | 0.447 |
| MeanShift(3) | сырые | LORD++ | 0.473 | 0.49 | 0.457 |
| MeanShift(1) | сырые | LORD++ | 0.361 | 0.423 | 0.284 |
| PH | сырые | uncorrected | 0.282 | 0.317 | 0.399 |
| MeanShift(3) | сырые | uncorrected | 0.219 | 0.245 | 0.322 |
| KS | сырые | uncorrected | 0.24 | 0.246 | 0.271 |
| MeanShift(1) | сырые | uncorrected | 0.133 | 0.136 | 0.18 |
| ADWIN | сырые | uncorrected | 0.135 | 0.139 | 0.132 |
| PH river по умолчанию | сырые | порог | 0.0538 | 0.0542 | 0.0591 |

### F1 по фактору `pattern`

| detector | view | rule | pattern = clustered | pattern = scattered |
|---|---|---|---|---|
| MeanShift(3) | остатки | bh_window | 0.896 | 0.871 |
| PH | остатки | bh_window | 0.882 | 0.873 |
| MeanShift(3) | остатки | bonferroni | 0.88 | 0.853 |
| PH | остатки | bonferroni | 0.856 | 0.868 |
| KS | остатки | bonferroni | 0.816 | 0.829 |
| KS | остатки | bh_window | 0.808 | 0.805 |
| MeanShift(1) | остатки | bonferroni | 0.78 | 0.794 |
| MeanShift(1) | остатки | bh_window | 0.78 | 0.79 |
| PH | сырые | bonferroni | 0.747 | 0.803 |
| PH | сырые | bh_window | 0.747 | 0.794 |
| MeanShift(3) | сырые | bonferroni | 0.724 | 0.799 |
| KS | сырые | bonferroni | 0.731 | 0.791 |
| MeanShift(3) | сырые | bh_window | 0.712 | 0.796 |
| KS | сырые | bh_window | 0.719 | 0.771 |
| MeanShift(1) | сырые | bonferroni | 0.678 | 0.739 |
| ADWIN | остатки | bonferroni | 0.666 | 0.679 |
| ADWIN | сырые | bonferroni | 0.641 | 0.667 |
| MeanShift(1) | сырые | bh_window | 0.672 | 0.611 |
| ADWIN | остатки | bh_window | 0.638 | 0.64 |
| ADWIN | сырые | LORD++ | 0.6 | 0.67 |
| ADWIN | сырые | bh_window | 0.581 | 0.6 |
| KS | сырые | LORD++ | 0.503 | 0.541 |
| PH | сырые | LORD++ | 0.475 | 0.483 |
| MeanShift(3) | сырые | LORD++ | 0.468 | 0.48 |
| MeanShift(1) | сырые | LORD++ | 0.363 | 0.349 |
| PH | сырые | uncorrected | 0.355 | 0.311 |
| MeanShift(3) | сырые | uncorrected | 0.284 | 0.24 |
| KS | сырые | uncorrected | 0.249 | 0.256 |
| MeanShift(1) | сырые | uncorrected | 0.163 | 0.136 |
| ADWIN | сырые | uncorrected | 0.138 | 0.133 |
| PH river по умолчанию | сырые | порог | 0.0574 | 0.0539 |
