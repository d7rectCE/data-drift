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

import math
from abc import ABC, abstractmethod

import numpy as np
from scipy import signal, stats


class Detector(ABC):
    """Base class: a drift detector exposed as a continuous, vectorised score.

    Subclasses implement ``scores`` (sequential detectors, whose statistic for a
    window is the maximum score inside it) or override ``window_statistics``
    directly (window tests such as KS or MeanShift), and give the score threshold
    that reproduces the detector's usual default as ``default_threshold``.
    """

    name: str = "detector"
    input_kind: str = "values"
    """Which scenario signal the detector consumes: ``values`` or ``errors``."""

    @property
    @abstractmethod
    def default_threshold(self) -> float:
        """Score threshold equivalent to the river default hyperparameters."""

    def scores(self, x: np.ndarray) -> np.ndarray:
        """Running score of shape ``(n_series, n_steps)`` for input of the same shape.

        The score at step ``t`` depends only on data up to ``t``; the detector would
        fire at ``t`` with sensitivity ``theta`` iff the score exceeds ``h(theta)``.
        """
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
        """Asymptotic KS p-value assuming i.i.d. data (what ``scipy.stats.ks_2samp`` reports)."""
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


def ar_whiten(x: np.ndarray, n_ref: int, order: int = 2) -> np.ndarray:
    """Standardised innovations of an AR(``order``) model fitted to each row's reference.

    For every row the mean, the Yule–Walker AR coefficients and the innovation
    standard deviation are estimated on the first ``n_ref`` steps only; the whole
    row (reference included) is then filtered: ``e_t = d_t - sum_i a_i d_{t-i}``
    with ``d = x - mean``, divided by the reference innovation sd. Under the null
    the output is close to white noise with unit variance, so a detector that
    assumes independent observations accumulates evidence at the right rate.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    d = x - x[:, :n_ref].mean(axis=1, keepdims=True)
    ref = d[:, :n_ref]
    p = max(0, min(order, n_ref // 4))
    gamma = np.stack([(ref[:, k:] * ref[:, : n_ref - k]).sum(axis=1) / n_ref for k in range(p + 1)], axis=1)
    e = d.copy()
    if p > 0:
        idx = np.abs(np.arange(p)[:, None] - np.arange(p)[None, :])
        R = gamma[:, idx] + 1e-12 * np.eye(p)
        a = np.linalg.solve(R, gamma[:, 1 : p + 1, None])[:, :, 0]
        a = np.where(gamma[:, :1] > 0, a, 0.0)
        for i in range(1, p + 1):
            e[:, i:] -= a[:, i - 1 : i] * d[:, :-i]
    sd = e[:, p:n_ref].std(axis=1, keepdims=True)
    return e / np.where(sd > 0, sd, 1.0)


class Prewhitened(Detector):
    """Another detector run on AR-whitened, standardised values (see ``ar_whiten``).

    Detectors such as Page-Hinkley treat consecutive observations as independent;
    on an autocorrelated error series calibration fixes their false-alarm rate but
    not the rate at which they accumulate evidence. The AR model is fitted on each
    reference (and on each bootstrap reference during calibration), so the p-values
    stay valid. Not for detectors of 0/1 errors (DDM).
    """

    def __init__(self, detector: Detector, order: int = 2):
        if detector.input_kind == "errors":
            raise ValueError(f"{detector.name} needs 0/1 errors and cannot run on whitened values")
        self.detector = detector
        self.order = order
        self.name = f"{detector.name}+AR"

    @property
    def default_threshold(self) -> float:
        """The inner detector's threshold (in units of the standardised innovations)."""
        return self.detector.default_threshold

    def window_statistics(self, x: np.ndarray, n_ref: int, window: int) -> np.ndarray:
        return self.detector.window_statistics(ar_whiten(x, n_ref, self.order), n_ref, window)

    def __repr__(self) -> str:
        return f"Prewhitened({self.detector!r}, order={self.order})"


class ECUSUM(Detector):
    """A sequential e-detector for a rise in the mean, in CUSUM form.

    On the AR-whitened, standardised innovations ``z_t`` (``ar_whiten``), each
    ``lambda`` in ``lambdas`` gives per-step e-values ``exp(lambda z_t - lambda^2 / 2)``
    and the CUSUM e-detector ``C_t = max(0, C_{t-1} + lambda z_t - lambda^2 / 2)`` (in
    logs); the score is the log of their average, ``log mean_k exp(C_t^(k))``, which
    covers shifts of unknown size (Shin, Ramdas & Rinaldo, 2023; multi-stream use as
    in Dandapanthula & Ramdas). The CUSUM starts at zero right after the reference.

    With exactly Gaussian white innovations an alarm at ``score >= log A`` would have
    an average run length of at least ``A`` under the null (``default_threshold`` uses
    ``A = 100``). Real innovations are not exactly Gaussian and the AR model is
    estimated, so driftfdr calibrates the threshold by bootstrap like any other
    statistic. Unlike the window tests, it can be checked at every step: see
    ``CalibratedDetector(sequential=True)``.
    """

    name = "e-CUSUM"

    def __init__(self, order: int = 2, lambdas=(0.25, 0.5, 1.0)):
        self.order = order
        self.lambdas = tuple(float(v) for v in lambdas)

    @property
    def default_threshold(self) -> float:
        return float(np.log(100.0))

    def _window_scores(self, x: np.ndarray, n_ref: int) -> np.ndarray:
        z = ar_whiten(x, n_ref, self.order)[:, n_ref:]
        logs = []
        for lam in self.lambdas:
            walk = np.cumsum(lam * z - lam * lam / 2.0, axis=1)
            low = np.minimum(np.minimum.accumulate(walk, axis=1), 0.0)
            logs.append(walk - low)
        stacked = np.stack(logs)
        top = stacked.max(axis=0)
        return top + np.log(np.exp(stacked - top).mean(axis=0))

    def start_stream(self, reference) -> "ECUSUMState":
        """Incremental state after ``reference``: ``state.update(x)`` returns the score at each new step."""
        ref = np.asarray(reference, dtype=float)
        n_ref = ref.size
        mu = ref.mean()
        d = ref - mu
        p = max(0, min(self.order, n_ref // 4))
        gamma = np.array([(d[k:] * d[: n_ref - k]).sum() / n_ref for k in range(p + 1)])
        a = np.zeros(p)
        if p > 0 and gamma[0] > 0:
            idx = np.abs(np.arange(p)[:, None] - np.arange(p)[None, :])
            a = np.linalg.solve(gamma[idx] + 1e-12 * np.eye(p), gamma[1 : p + 1])
        e = d.copy()
        for i in range(1, p + 1):
            e[i:] -= a[i - 1] * d[:-i]
        sd = e[p:].std()
        return ECUSUMState(mu=float(mu), coef=[float(v) for v in a], sd=float(sd) if sd > 0 else 1.0,
                           lags=[float(v) for v in d[::-1][:p]], lambdas=list(self.lambdas),
                           cusums=[0.0] * len(self.lambdas))

    def load_stream(self, data: dict) -> "ECUSUMState":
        """Restore a state saved with ``ECUSUMState.to_dict``."""
        return ECUSUMState.from_dict(data)

    def __repr__(self) -> str:
        return f"ECUSUM(order={self.order}, lambdas={self.lambdas})"


class ECUSUMState:
    """Running state of an ``ECUSUM`` after its reference; plain floats, so updates are cheap."""

    __slots__ = ("mu", "coef", "sd", "lags", "lambdas", "cusums")

    def __init__(self, mu, coef, sd, lags, lambdas, cusums):
        self.mu, self.coef, self.sd = mu, coef, sd
        self.lags, self.lambdas, self.cusums = lags, lambdas, cusums

    def update(self, x: float) -> float:
        """Add one observation; return the current score (log of the mixture e-detector)."""
        d = x - self.mu
        e = d
        lags = self.lags
        for a, prev in zip(self.coef, lags):
            e -= a * prev
        if lags:
            lags.insert(0, d)
            lags.pop()
        z = e / self.sd
        cusums, top, total = self.cusums, 0.0, 0.0
        for k, lam in enumerate(self.lambdas):
            c = cusums[k] + lam * z - 0.5 * lam * lam
            if c < 0.0:
                c = 0.0
            cusums[k] = c
            if c > top:
                top = c
        for c in cusums:
            total += math.exp(c - top)
        return top + math.log(total / len(cusums))

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}

    @classmethod
    def from_dict(cls, data: dict) -> "ECUSUMState":
        return cls(**data)


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
    """The four detectors of the original study, with river-default thresholds: PH, DDM, ADWIN, KS."""
    return [PageHinkley(), DDM(), ADWIN(), KSWindow(n_ref=n_ref, window=window)]
