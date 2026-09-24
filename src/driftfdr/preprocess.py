"""Turning raw per-event errors into monitorable series, and choosing the tolerance.

Monitoring works on one value per model per step. Real systems log errors at
irregular times (many predictions in a busy hour, none at night), and many
signals have daily or weekly cycles. ``bucket_means`` averages events into
fixed time buckets, so a step is an hour or a day rather than a row; with a
window of one full cycle, the cycle itself cancels out (see experiment 16).
"""

from __future__ import annotations

import numpy as np


def bucket_means(values: np.ndarray, bucket: np.ndarray, n_buckets: int | None = None) -> np.ndarray:
    """Mean of ``values`` (shape ``(n_events,)`` or ``(n_series, n_events)``) per integer bucket id.

    Empty buckets are filled with the previous bucket's mean (the first one with the
    overall mean), so the output has no gaps; returns shape ``(n_series, n_buckets)``.
    """
    values = np.atleast_2d(np.asarray(values, dtype=float))
    bucket = np.asarray(bucket, dtype=np.int64)
    n_buckets = int(bucket.max()) + 1 if n_buckets is None else n_buckets
    counts = np.bincount(bucket, minlength=n_buckets).astype(float)
    sums = np.stack([np.bincount(bucket, weights=row, minlength=n_buckets) for row in values])
    out = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    for j in range(n_buckets):
        if counts[j] == 0:
            out[:, j] = out[:, j - 1] if j else values.mean(axis=1)
    return out


def tolerance_from_cost(retrain_cost: float, horizon: float) -> float:
    """Smallest rise of the per-step loss that is worth one retraining.

    A drifted model left alone costs ``delta`` extra loss per step for ``horizon``
    steps (until the next scheduled retraining, or the planning horizon); a
    retraining costs ``retrain_cost`` in the same loss units. Retraining pays off
    when ``delta * horizon > retrain_cost``, so the tolerance of the
    material-degradation null is ``retrain_cost / horizon``.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return retrain_cost / horizon
