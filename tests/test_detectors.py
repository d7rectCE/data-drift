import numpy as np
import pytest
from scipy import stats

from driftfdr.detectors import ADWIN, DDM, KSWindow, PageHinkley, ks_statistic


def _first_river_alarm(detector, xs):
    for i, x in enumerate(xs):
        detector.update(x)
        if detector.drift_detected:
            return i
    return None


def _first_alarm(scores, threshold):
    hits = np.flatnonzero(scores > threshold)
    return int(hits[0]) if hits.size else None


@pytest.mark.parametrize("mode", ["up", "down", "both"])
def test_page_hinkley_matches_river(mode):
    drift = pytest.importorskip("river.drift")
    rng = np.random.default_rng(0)
    for _ in range(30):
        shift = rng.uniform(0.3, 1.5) * (1 if mode != "down" else -1)
        x = np.concatenate([rng.normal(0, 1, 600), rng.normal(shift, 1, 600)])
        river_alarm = _first_river_alarm(drift.PageHinkley(mode=mode), x)
        ours = _first_alarm(PageHinkley(mode=mode).scores(x)[0], 50.0)
        assert ours == river_alarm


def test_ddm_matches_river():
    binary = pytest.importorskip("river.drift.binary")
    rng = np.random.default_rng(1)
    for _ in range(100):
        x = np.concatenate([rng.random(500) < 0.2, rng.random(500) < rng.uniform(0.2, 0.6)])
        x = x.astype(int)
        river_alarm = _first_river_alarm(binary.DDM(), x)
        ours = _first_alarm(DDM().scores(x)[0], 3.0)
        assert ours == river_alarm


def test_adwin_close_to_river():
    """River only checks for cuts every ``clock=32`` steps, so ours may fire earlier."""
    drift = pytest.importorskip("river.drift")
    rng = np.random.default_rng(2)
    for _ in range(10):
        x = np.concatenate([rng.normal(0, 1, 1000), rng.normal(1.0, 1, 500)])
        river_alarm = _first_river_alarm(drift.ADWIN(), x)
        ours = _first_alarm(ADWIN().scores(x)[0], ADWIN().default_threshold)
        assert river_alarm is not None and ours is not None
        assert river_alarm - 64 <= ours <= river_alarm + 32


def test_adwin_window_statistic_matches_full_scores():
    x = np.random.default_rng(3).normal(size=(4, 400))
    det = ADWIN()
    assert np.allclose(det.window_statistic(x, 300), det.scores(x)[:, 300:].max(axis=1))


def test_score_is_monotone_in_sensitivity():
    """A lower threshold (more sensitive detector) never fires later."""
    x = np.random.default_rng(4).normal(size=(1, 800))
    x[0, 400:] += 0.7
    s = PageHinkley().scores(x)[0]
    times = [_first_alarm(s, h) for h in (10, 20, 40)]
    assert times[0] <= times[1] <= times[2]


def test_ks_statistic_matches_scipy_with_ties():
    rng = np.random.default_rng(5)
    a = rng.integers(0, 10, size=(20, 60)).astype(float)
    b = rng.integers(0, 12, size=(20, 25)).astype(float)
    ours = ks_statistic(a, b)
    ref = [stats.ks_2samp(a[i], b[i]).statistic for i in range(20)]
    assert np.allclose(ours, ref)


def test_ks_default_threshold_is_nominal_level():
    det = KSWindow(alpha=0.05, n_ref=300, window=100)
    p = det.nominal_pvalue(det.default_threshold, 300, 100)
    assert p == pytest.approx(0.05, rel=1e-6)
