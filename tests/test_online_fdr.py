import numpy as np
import pytest
from scipy import stats

from driftfdr.online_fdr import (
    LOND,
    SAFFRON,
    AlphaInvesting,
    BHWindow,
    LORDpp,
    Uncorrected,
    benjamini_hochberg,
    gamma_lord,
    gamma_saffron,
)


def test_gamma_sequences_sum_to_one():
    j = np.arange(1, 2_000_001)
    assert gamma_saffron(j).sum() == pytest.approx(1.0, abs=0.01)
    # the LORD sequence has a very heavy tail; the partial sum approaches 1 slowly
    assert 0.5 < gamma_lord(j).sum() < 1.0
    assert np.all(np.diff(gamma_lord(j[:1000])) <= 0)


def test_benjamini_hochberg_is_step_up():
    p = np.array([0.005, 0.035, 0.03, 0.9])
    # thresholds 0.0125, 0.025, 0.0375, 0.05 for sorted [0.005, 0.03, 0.035, 0.9]:
    # 0.03 fails its own threshold but is rejected because 0.035 passes
    assert benjamini_hochberg(p, 0.05).tolist() == [True, True, True, False]
    assert BHWindow(0.05).decide(p, None).sum() == 3
    assert not benjamini_hochberg(np.array([0.2, 0.6]), 0.05).any()


def test_lordpp_first_level_and_wealth_after_rejection():
    proc = LORDpp(alpha=0.1)
    assert proc.test(1e-9)
    assert proc.levels[0] == pytest.approx(0.05 * gamma_lord(1))
    proc.test(0.9)
    expected = 0.05 * gamma_lord(2) + 0.05 * gamma_lord(1)
    assert proc.levels[1] == pytest.approx(expected)


def test_saffron_levels_bounded_by_lambda():
    proc = SAFFRON(alpha=0.2, lam=0.5)
    rng = np.random.default_rng(0)
    for p in rng.uniform(size=200) ** 3:
        proc.test(p)
    assert max(proc.levels) <= 0.5


def test_alpha_investing_wealth_stays_nonnegative():
    proc = AlphaInvesting(alpha=0.05)
    for p in np.random.default_rng(1).uniform(size=500):
        proc.test(p)
        assert proc.wealth >= 0


def _simulate_fdr(factory, n_reps=200, n=400, pi1=0.1, seed=0):
    rng = np.random.default_rng(seed)
    fdps, powers = [], []
    for _ in range(n_reps):
        alt = rng.random(n) < pi1
        z = rng.normal(size=n) + 3.0 * alt
        p = stats.norm.sf(z)
        rej = factory().decide(p, rng)
        fdps.append((rej & ~alt).sum() / max(rej.sum(), 1))
        powers.append((rej & alt).sum() / max(alt.sum(), 1))
    return np.mean(fdps), np.mean(powers)


@pytest.mark.parametrize("cls", [LOND, LORDpp, SAFFRON, BHWindow])
def test_fdr_controlled_on_independent_pvalues(cls):
    fdr, power = _simulate_fdr(lambda: cls(0.1))
    assert fdr <= 0.1 + 0.02
    assert power > 0.05


def test_uncorrected_does_not_control_fdr():
    fdr, _ = _simulate_fdr(lambda: Uncorrected(0.1), pi1=0.02)
    assert fdr > 0.3
