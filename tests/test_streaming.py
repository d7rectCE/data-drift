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


def _feed(mon, values, start, stop, ids=None):
    alarms = []
    for t in range(start, stop):
        obs = {m: values[i, t] for i, m in enumerate(ids)} if ids else values[:, t]
        out = mon.update(obs)
        alarms += [(m, t) for m in out]
    return alarms


def test_save_and_load_continue_identically(tmp_path):
    sc = make_scenario(ScenarioConfig(n_streams=6, n_steps=1400, phi=0.3, drift_fraction=0.5, magnitude=1.5), seed=5)
    kw = dict(procedure="bh_window", alpha=0.2, n_ref=200, window=100, horizon=3, calibration=CAL, seed=1)
    ids = ["a", "b", "c", "d", "e", "f"]
    straight = StreamingMonitor(detector_factory=PageHinkley, model_ids=ids, **kw)
    first = _feed(straight, sc.values, 0, 700, ids)
    rest = _feed(straight, sc.values, 700, 1400, ids)

    resumed = StreamingMonitor(detector_factory=PageHinkley, model_ids=ids, **kw)
    assert _feed(resumed, sc.values, 0, 700, ids) == first
    resumed.save(tmp_path / "state.npz")
    loaded = StreamingMonitor.load(tmp_path / "state.npz", detector_factory=PageHinkley)
    assert _feed(loaded, sc.values, 700, 1400, ids) == rest
    assert len(first + rest) > 0


def test_models_can_be_added_removed_and_report_irregularly():
    rng = np.random.default_rng(3)
    mon = StreamingMonitor(detector_factory=PageHinkley, model_ids=["x"], n_ref=200, window=100, horizon=2,
                           calibration=CAL)
    for _ in range(150):
        mon.update({"x": rng.normal()})
    mon.add_model("y")
    for t in range(400):
        obs = {"x": rng.normal()}
        if t % 2 == 0:  # y reports every other step
            obs["y"] = rng.normal()
        mon.update(obs)
    assert mon.n_seen == {"x": 550, "y": 200}
    assert mon.models["x"].calibrated and mon.models["y"].calibrated
    mon.remove_model("x")
    assert list(mon.models) == ["y"]
    with pytest.raises(KeyError):
        mon.update({"x": 0.0})
