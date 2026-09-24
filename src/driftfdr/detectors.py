"""Drift detectors rewritten as continuous, vectorised scores.

A practical detector fires when an internal statistic crosses a threshold set
by its sensitivity hyperparameter (``lambda`` for Page-Hinkley, the number of
standard deviations for DDM, ``delta`` for ADWIN, ``alpha`` for KS). Each class
below exposes that statistic as a *score* on a common scale, so that

    detector with hyperparameter theta fires at time t  <=>  score_t > h(theta).

The score is the "critical sensitivity" at which the detector would have
fired, which is what makes it calibratable into a p-value: the binary alarm is
recovered by thresholding, and ``default_threshold`` reproduces the river
default. Page-Hinkley and DDM reproduce river's formulas exactly up to the
first alarm (see ``tests/test_detectors.py``); ADWIN uses river's cut criterion
on a fixed geometric grid of split points instead of the exponential histogram.

All ``scores`` methods take an array of shape ``(n_series, n_steps)`` and
return an array of the same shape, so thousands of bootstrap replicates are
scored in one call. Scores are causal: the score at step ``t`` only uses data
up to ``t``, so one pass over a long series yields the statistics of all its
prefixes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy import signal, stats


class Detector(ABC):
    name: str = "detector"
    input_kind: str = "values"
    """Which scenario signal the detector consumes: ``values`` or ``errors``."""

    @property
    @abstractmethod
    def default_threshold(self) -> float:
        """Score threshold equivalent to the river default hyperparameters."""

    def scores(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def window_statistics(self, x: np.ndarray, n_ref: int, window: int) -> np.ndarray:
        """Test statistics for "data after the reference differ from it".

        ``x`` holds the reference in its first ``n_ref`` columns followed by
        ``n`` windows of ``window`` steps. The detector is warmed up on the
        reference and run over the windows; column ``i`` of the result is the
        maximum score reached inside window ``i``, shape ``(n_series, n)``.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n = (x.shape[1] - n_ref) // window
        s = self._window_scores(x, n_ref)[:, : n * window]
        return s.reshape(x.shape[0], n, window).max(axis=2)

    def _window_scores(self, x: np.ndarray, n_ref: int) -> np.ndarray:
        return self.scores(x)[:, n_ref:]

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class PageHinkley(Detector):
    """Page-Hinkley test as in ``river.drift.PageHinkley``; score is compared with ``lambda``."""

    name = "PH"

    def __init__(
        self,
        delta: float = 0.005,
        alpha: float = 1 - 0.0001,
        min_instances: int = 30,
        mode: str = "up",
        threshold: float = 50.0,
    ):
        if mode not in ("up", "down", "both"):
            raise ValueError("mode must be 'up', 'down' or 'both'")
        self.delta = delta
        self.alpha = alpha
        self.min_instances = min_instances
        self.mode = mode
        self.threshold = threshold

    @property
    def default_threshold(self) -> float:
        return self.threshold

    def scores(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n = np.arange(1, x.shape[1] + 1)
        dev = x - np.cumsum(x, axis=1) / n
        fading = ([1.0], [1.0, -self.alpha])
        score = np.zeros_like(x)
        if self.mode in ("up", "both"):
            inc = signal.lfilter(*fading, dev - self.delta, axis=1)
            score = inc - np.minimum.accumulate(inc, axis=1)
        if self.mode in ("down", "both"):
            dec = signal.lfilter(*fading, dev + self.delta, axis=1)
            # river initialises the running maximum at -1
            test_dec = np.maximum(np.maximum.accumulate(dec, axis=1), -1.0) - dec
            score = np.maximum(score, test_dec)
        score[:, : self.min_instances - 1] = 0.0
        return score


class DDM(Detector):
    """Drift Detection Method as in ``river.drift.binary.DDM``.

    Score is ``(p_t + s_t - p_min) / s_min``; river fires when it exceeds
    ``drift_threshold`` (3 by default).
    """

    name = "DDM"
    input_kind = "errors"

    def __init__(self, warm_start: int = 30, drift_threshold: float = 3.0):
        self.warm_start = warm_start
        self.drift_threshold = drift_threshold

    @property
    def default_threshold(self) -> float:
        return self.drift_threshold

    def scores(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        pos = np.arange(x.shape[1])
        n = pos + 1
        p = np.cumsum(x, axis=1) / n
        s = np.sqrt(p * (1 - p) / n)
        ps = p + s
        valid = n > self.warm_start
        ps_valid = np.where(valid, ps, np.inf)
        # river updates the minimum on ties (<=), so track the latest argmin
        is_min = valid & (ps_valid <= np.minimum.accumulate(ps_valid, axis=1))
        arg = np.maximum.accumulate(np.where(is_min, pos, -1), axis=1)
        has_min = arg >= 0
        arg = np.maximum(arg, 0)
        p_min = np.take_along_axis(p, arg, axis=1)
        s_min = np.take_along_axis(s, arg, axis=1)
        excess = ps - p_min
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.where(s_min > 0, excess / s_min, np.where(excess > 0, np.inf, 0.0))
        return np.where(has_min, score, 0.0)


class ADWIN(Detector):
    """ADWIN-style statistic with river's cut criterion.

    For a window ``W = W0 + W1`` of width ``n`` river cuts when
    ``|mu0 - mu1| > eps``, ``eps = sqrt(2 m v L) + 2/3 m L``, where
    ``L = ln(2 ln n / delta)``, ``v`` is the window variance and ``m`` the
    harmonic term of the two sub-window sizes. Solving for the smallest
    ``delta`` that produces a cut gives the score ``-ln(delta*)``; river fires
    when it exceeds ``-ln(delta)``. The window grows from the start of the
    reference (no shrinking) and split points lie on a geometric grid.
    """

    name = "ADWIN"

    def __init__(
        self,
        delta: float = 0.002,
        min_window_length: int = 5,
        grace_period: int = 10,
        n_splits: int = 24,
    ):
        self.delta = delta
        self.min_window_length = min_window_length
        self.grace_period = grace_period
        self.n_splits = n_splits

    @property
    def default_threshold(self) -> float:
        return -np.log(self.delta)

    def scores(self, x: np.ndarray) -> np.ndarray:
        return self._scores_from(np.atleast_2d(np.asarray(x, dtype=float)), 1)

    def _window_scores(self, x: np.ndarray, n_ref: int) -> np.ndarray:
        return self._scores_from(x, n_ref + 1)

    def _scores_from(self, x: np.ndarray, first_width: int) -> np.ndarray:
        """Scores for window widths ``first_width..T``; column ``i`` is width ``first_width + i``."""
        T = x.shape[1]
        mwl = self.min_window_length
        C = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x, axis=1)], axis=1)
        Q = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x * x, axis=1)], axis=1)
        width = np.arange(first_width, T + 1)
        mean = C[:, first_width:] / width
        var = np.maximum(Q[:, first_width:] / width - mean**2, 0.0)
        log_term = np.log(2 * np.log(np.maximum(width, 2)))
        best = np.zeros((x.shape[0], width.size))
        sizes = np.unique(np.geomspace(mwl, max(mwl, T - mwl), self.n_splits).astype(int))
        for n1 in sizes:
            # widths where both sub-windows are long enough and the grace period is over
            lo = max(first_width, n1 + mwl, self.grace_period + 1)
            if lo > T:
                continue
            cols = slice(lo - first_width, None)
            n0 = np.arange(lo, T + 1) - n1
            c_split = C[:, lo - n1 : T + 1 - n1]
            mu1 = (C[:, lo:] - c_split) / n1
            mu0 = c_split / n0
            m = 1.0 / (n0 - mwl + 1) + 1.0 / (n1 - mwl + 1)
            a = (2.0 / 3.0) * m
            b = np.sqrt(2.0 * m * var[:, cols])
            u = (np.sqrt(b * b + 4.0 * a * np.abs(mu0 - mu1)) - b) / (2.0 * a)
            np.maximum(best[:, cols], u * u - log_term[cols], out=best[:, cols])
        return best


class KSWindow(Detector):
    """Two-sample Kolmogorov–Smirnov statistic between the reference and recent data.

    The statistic for window ``i`` compares the reference with everything from
    the end of the reference up to the end of that window. The default
    threshold is the asymptotic 5% critical value for a single window under
    i.i.d. observations, i.e. what ``scipy.stats.ks_2samp`` would use.
    """

    name = "KS"

    def __init__(self, alpha: float = 0.05, n_ref: int = 300, window: int = 100):
        self.alpha = alpha
        self.n_ref = n_ref
        self.window = window

    @property
    def default_threshold(self) -> float:
        en = round(self.n_ref * self.window / (self.n_ref + self.window))
        return float(stats.kstwo.isf(self.alpha, en))

    def nominal_pvalue(self, statistic: np.ndarray, n_ref: int, window: int) -> np.ndarray:
        en = round(n_ref * window / (n_ref + window))
        return stats.kstwo.sf(statistic, en)

    def window_statistics(self, x: np.ndarray, n_ref: int, window: int) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n = (x.shape[1] - n_ref) // window
        ref = x[:, :n_ref]
        return np.stack(
            [ks_statistic(ref, x[:, n_ref : n_ref + (i + 1) * window]) for i in range(n)], axis=1
        )


class KSSliding(Detector):
    """Sequential KS in the spirit of IKS (dos Reis et al., 2016).

    The reference is compared with a sliding window of the last ``width``
    observations every ``stride`` steps; the statistic of a check window is the
    largest KS distance reached inside it. Unlike ``KSWindow`` it forgets old
    data, so a short-lived change is not diluted by a long look-back.
    """

    name = "KS-sliding"

    def __init__(self, width: int = 100, stride: int = 10, alpha: float = 0.05, n_ref: int = 300):
        self.width, self.stride, self.alpha, self.n_ref = width, stride, alpha, n_ref

    @property
    def default_threshold(self) -> float:
        en = round(self.n_ref * self.width / (self.n_ref + self.width))
        return float(stats.kstwo.isf(self.alpha, en))

    def window_statistics(self, x: np.ndarray, n_ref: int, window: int) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n = (x.shape[1] - n_ref) // window
        ref = x[:, :n_ref]
        out = np.zeros((x.shape[0], n))
        for i in range(n):
            for end in range(n_ref + i * window + self.stride, n_ref + (i + 1) * window + 1, self.stride):
                start = max(n_ref, end - self.width)
                out[:, i] = np.maximum(out[:, i], ks_statistic(ref, x[:, start:end]))
        return out


class MeanShift(Detector):
    """Persistent rise of the mean: the smallest window-mean excess over the last ``persistence`` windows.

    The statistic for window ``i`` is ``min_{j in last m windows} mean(window j) - mean(reference)``
    where windows not yet observed count as zero excess, so no alarm is possible before
    ``persistence`` windows have passed. A spike confined to one window cannot make it
    large when ``persistence > 1``, so alarms require the rise to last. With
    ``persistence = 1`` it is the plain window-vs-reference mean test used by Rombouts &
    Wilms for loss monitoring. There is no river counterpart; ``default_threshold`` is
    a fixed rise of 0.1 in signal units.
    """

    name = "MeanShift"

    def __init__(self, persistence: int = 3):
        self.persistence = persistence

    @property
    def default_threshold(self) -> float:
        return 0.1

    def window_statistics(self, x: np.ndarray, n_ref: int, window: int) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n = (x.shape[1] - n_ref) // window
        ref = x[:, :n_ref].mean(axis=1, keepdims=True)
        means = x[:, n_ref : n_ref + n * window].reshape(x.shape[0], n, window).mean(axis=2) - ref
        out = np.empty_like(means)
        for i in range(n):
            recent = means[:, max(0, i - self.persistence + 1) : i + 1].min(axis=1)
            out[:, i] = recent if i + 1 >= self.persistence else np.minimum(recent, 0.0)
        return out

    def __repr__(self) -> str:
        return f"MeanShift(persistence={self.persistence})"


def ks_statistic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise two-sample KS statistic, correct in the presence of ties."""
    n, m = a.shape[1], b.shape[1]
    z = np.concatenate([a, b], axis=1)
    w = np.concatenate([np.full(n, 1.0 / n), np.full(m, -1.0 / m)])
    order = np.argsort(z, axis=1, kind="stable")
    zs = np.take_along_axis(z, order, axis=1)
    cdf_gap = np.abs(np.cumsum(w[order], axis=1))
    end_of_tie = np.ones_like(zs, dtype=bool)
    end_of_tie[:, :-1] = zs[:, 1:] != zs[:, :-1]
    return np.where(end_of_tie, cdf_gap, 0.0).max(axis=1)


def default_detectors(n_ref: int = 300, window: int = 100) -> list[Detector]:
    return [PageHinkley(), DDM(), ADWIN(), KSWindow(n_ref=n_ref, window=window)]
