import numpy as np
import pytest

from driftfdr import (
    CalibratedDetector,
    CalibrationConfig,
    MonitorConfig,
    PageHinkley,
    ScenarioConfig,
    StreamingMonitor,
    from_river,
    make_procedure,
    make_scenario,
    run_monitor,
)

CAL = CalibrationConfig(n_boot=100, method="sieve")


def test_streaming_monitor_reproduces_batch_run():
    sc = make_scenario(ScenarioConfig(n_streams=12, n_steps=1500, phi=0.3, drift_fraction=0.25, magnitude=1.5), seed=4)
    batch = run_monitor(
        sc, PageHinkley(), make_procedure("uncorrected", 0.1),
        MonitorConfig(n_ref=200, window=100, horizon=3, calibration=CAL), seed=0,
    ).tests
    mon = StreamingMonitor(12, PageHinkley, "uncorrected", 0.1, n_ref=200, window=100, horizon=3, calibration=CAL, seed=0)
    alarms = []
    for t in range(sc.n_steps):
        for k in mon.update(sc.values[:, t]):
            alarms.append((int(k), t + 1))
    expected = sorted(zip(batch.loc[batch.rejected, "stream"], batch.loc[batch.rejected, "t_end"]))
    assert sorted(alarms) == [(int(k), int(t)) for k, t in expected]
    assert len(alarms) > 0


def test_calibrated_detector_emits_pvalue_once_per_window():
    det = CalibratedDetector(PageHinkley(), n_ref=200, window=50, horizon=2, calibration=CAL, seed=1)
    x = np.random.default_rng(0).normal(size=400)
    out = [det.update(v) for v in x]
    ps = [p for p in out if p is not None]
    assert len(ps) == 4 and all(0 < p <= 1 for p in ps)
    assert all(p is None for p in out[:249])
    det.reset()
    assert not det.calibrated


def test_from_river_copies_hyperparameters():
    drift = pytest.importorskip("river.drift")
    ph = from_river(drift.PageHinkley(threshold=30, delta=0.01, mode="up"))
    assert ph.default_threshold == 30 and ph.delta == 0.01
    assert from_river(drift.ADWIN(delta=0.01)).default_threshold == pytest.approx(-np.log(0.01))
    assert from_river(drift.binary.DDM(drift_threshold=2.5)).default_threshold == 2.5
