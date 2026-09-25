"""Label-free error estimates from NannyML's CBPE, as a signal to monitor."""

from __future__ import annotations

import warnings

import numpy as np


def cbpe_estimated_error(reference, analysis, chunk_size: int, y_pred_proba="y_pred_proba", y_pred="y_pred",
                         y_true="y_true", problem_type: str = "classification_binary") -> np.ndarray:
    """Estimated error rate per chunk from NannyML's confidence-based performance estimation.

    CBPE is fitted on ``reference`` (a DataFrame with predictions, probabilities and
    labels) and estimates the accuracy of every ``chunk_size`` rows of ``analysis``
    without labels. Returns ``1 - estimated accuracy`` per chunk: one step of a series to
    feed to a ``StreamingMonitor`` while labels are delayed.

    CBPE assumes calibrated probabilities and no change in p(y|X): it cannot see real
    concept drift, only shifts in the inputs that make the model less confident
    (experiment 11 shows that the former is the harmful kind). Requires ``nannyml``.
    """
    import nannyml as nml

    estimator = nml.CBPE(y_pred_proba=y_pred_proba, y_pred=y_pred, y_true=y_true, metrics=["accuracy"],
                         chunk_size=chunk_size, problem_type=problem_type)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(reference)
        result = estimator.estimate(analysis).filter(period="analysis").to_df()
    return 1.0 - result[("accuracy", "value")].to_numpy(dtype=float)
