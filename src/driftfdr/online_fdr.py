"""Decision rules applied to the p-values of one monitoring window.

Every window produces a batch of p-values, one per actively monitored stream.
Batch rules (``Uncorrected``, ``BonferroniWindow``, ``BHWindow``) decide on the
batch as a whole. Online rules (``LOND``, ``LORDpp``, ``SAFFRON``,
``AlphaInvesting``) see hypotheses one at a time; within a batch they are fed
in a random order that does not depend on the p-values.

References: Foster & Stine (2008), Javanmard & Montanari (2018),
Ramdas et al. (2017, LORD++), Ramdas et al. (2018, SAFFRON).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

_LORD_CONST = 0.07720838
_SAFFRON_CONST = 0.4374901658


def gamma_lord(j: np.ndarray) -> np.ndarray:
    """Javanmard–Montanari sequence, sums to one over j >= 1."""
    j = np.asarray(j, dtype=float)
    return _LORD_CONST * np.log(np.maximum(j, 2)) / (j * np.exp(np.sqrt(np.log(j))))


def gamma_saffron(j: np.ndarray) -> np.ndarray:
    """``gamma_j ∝ j^-1.6``, sums to one over j >= 1."""
    return _SAFFRON_CONST / np.asarray(j, dtype=float) ** 1.6


class Procedure(ABC):
    name = "procedure"
    uses_statistics = False
    """If true, ``decide`` receives raw detector statistics instead of p-values."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    @abstractmethod
    def decide(self, values: np.ndarray, rng) -> np.ndarray:
        """Boolean rejections for one window."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(alpha={self.alpha})"


class RawThreshold(Procedure):
    """Uncalibrated detector: alarm when the statistic crosses the default threshold."""

    name = "raw"
    uses_statistics = True

    def __init__(self, threshold: float):
        super().__init__(alpha=np.nan)
        self.threshold = threshold

    def decide(self, values, rng):
        return np.asarray(values) > self.threshold


class Uncorrected(Procedure):
    """Each stream tested at level alpha, as in current practice."""

    name = "uncorrected"

    def decide(self, values, rng):
        return np.asarray(values) <= self.alpha


class BonferroniWindow(Procedure):
    """Bonferroni within each window: controls P(any false alarm in a window)."""

    name = "bonferroni"

    def decide(self, values, rng):
        values = np.asarray(values)
        return values <= self.alpha / max(values.size, 1)


class BHWindow(Procedure):
    """Benjamini–Hochberg within each window."""

    name = "bh_window"

    def decide(self, values, rng):
        return benjamini_hochberg(values, self.alpha)


def benjamini_hochberg(p: np.ndarray, alpha: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    sorted_p = np.sort(p)
    below = np.flatnonzero(sorted_p <= alpha * np.arange(1, m + 1) / m)
    if below.size == 0:
        return np.zeros(m, dtype=bool)
    return p <= sorted_p[below[-1]]


class OnlineProcedure(Procedure):
    """Sequential rule: hypotheses ``t = 1, 2, ...`` each get a level ``alpha_t``."""

    def __init__(self, alpha: float = 0.05):
        super().__init__(alpha)
        self.t = 0
        self.levels: list[float] = []

    def decide(self, values, rng):
        values = np.asarray(values, dtype=float)
        rejected = np.zeros(values.size, dtype=bool)
        for i in rng.permutation(values.size):
            rejected[i] = self.test(values[i])
        return rejected

    def test(self, p: float) -> bool:
        self.t += 1
        level = self.next_level()
        self.levels.append(level)
        rejected = bool(p <= level)
        self.update(p, rejected)
        return rejected

    @abstractmethod
    def next_level(self) -> float: ...

    @abstractmethod
    def update(self, p: float, rejected: bool) -> None: ...


class LOND(OnlineProcedure):
    """``alpha_t = alpha * gamma_t * (D_{t-1} + 1)``; FDR control under PRDS."""

    name = "LOND"

    def __init__(self, alpha: float = 0.05):
        super().__init__(alpha)
        self.n_rejections = 0

    def next_level(self):
        return float(self.alpha * gamma_lord(self.t) * (self.n_rejections + 1))

    def update(self, p, rejected):
        self.n_rejections += rejected


class LORDpp(OnlineProcedure):
    """LORD++: wealth is earned back at every rejection."""

    name = "LORD++"

    def __init__(self, alpha: float = 0.05, w0: float | None = None):
        super().__init__(alpha)
        self.w0 = alpha / 2 if w0 is None else w0
        self.rejection_times: list[int] = []

    def next_level(self):
        t = self.t
        level = self.w0 * gamma_lord(t)
        if self.rejection_times:
            taus = np.asarray(self.rejection_times)
            g = gamma_lord(t - taus)
            level += (self.alpha - self.w0) * g[0] + self.alpha * g[1:].sum()
        return float(level)

    def update(self, p, rejected):
        if rejected:
            self.rejection_times.append(self.t)


class SAFFRON(OnlineProcedure):
    """SAFFRON: adapts to the fraction of nulls through candidates ``p <= lambda``."""

    name = "SAFFRON"

    def __init__(self, alpha: float = 0.05, lam: float = 0.5, w0: float | None = None):
        super().__init__(alpha)
        self.lam = lam
        self.w0 = (1 - lam) * alpha / 2 if w0 is None else w0
        self.n_candidates = 0
        self.rejection_times: list[int] = []
        self.candidates_at_rejection: list[int] = []

    def next_level(self):
        t = self.t
        level = self.w0 * gamma_saffron(t - self.n_candidates)
        if self.rejection_times:
            taus = np.asarray(self.rejection_times)
            after = self.n_candidates - np.asarray(self.candidates_at_rejection)
            g = gamma_saffron(t - taus - after)
            level += ((1 - self.lam) * self.alpha - self.w0) * g[0]
            level += (1 - self.lam) * self.alpha * g[1:].sum()
        return float(min(self.lam, level))

    def update(self, p, rejected):
        self.n_candidates += p <= self.lam
        if rejected:
            self.rejection_times.append(self.t)
            self.candidates_at_rejection.append(self.n_candidates)


class AlphaInvesting(OnlineProcedure):
    """Foster–Stine alpha-investing (mFDR control) with the ``W / (1 + t - k*)`` spending rule."""

    name = "alpha-investing"

    def __init__(self, alpha: float = 0.05, w0: float | None = None, payout: float | None = None):
        super().__init__(alpha)
        self.wealth = alpha / 2 if w0 is None else w0
        self.payout = alpha if payout is None else payout
        self.last_rejection = 0

    def next_level(self):
        return float(self.wealth / (1 + self.t - self.last_rejection))

    def update(self, p, rejected):
        level = self.levels[-1]
        if rejected:
            self.wealth += self.payout
            self.last_rejection = self.t
        else:
            self.wealth -= level / (1 - level)


PROCEDURES = {
    cls.name: cls
    for cls in (Uncorrected, BonferroniWindow, BHWindow, LOND, LORDpp, SAFFRON, AlphaInvesting)
}


def make_procedure(name: str, alpha: float = 0.05) -> Procedure:
    return PROCEDURES[name](alpha)
