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


def _common_factor_values(n=10, steps=1200, seed=0, shift_all_at=None):
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(size=steps)) * 0.05 + rng.normal(size=steps)
    x = 2.0 + 0.5 * np.arange(n)[:, None] + common + rng.normal(scale=0.5, size=(n, steps))
    if shift_all_at is not None:
        x[:, shift_all_at:] += 3.0
    return x


def test_split_common_matches_offline_residuals():
    from driftfdr import split_common

    x = _common_factor_values()
    kw = dict(procedure="bonferroni", alpha=0.05, n_ref=200, window=100, horizon=3, calibration=CAL, seed=2)
    split = StreamingMonitor(detector_factory=PageHinkley, n_models=10, split_common=True, **kw)
    resid, common = split_common(x, 200)
    plain = StreamingMonitor(detector_factory=PageHinkley, model_ids=[*range(10), StreamingMonitor.FLEET], **kw)
    n_windows = 0
    for t in range(x.shape[1]):
        a = split.update(x[:, t])
        b = plain.update({**{k: resid[k, t] for k in range(10)}, StreamingMonitor.FLEET: common[t]})
        assert split.last_pvalues.keys() == plain.last_pvalues.keys()
        for k, p in split.last_pvalues.items():
            assert p == pytest.approx(plain.last_pvalues[k], abs=1e-9)
        n_windows += bool(split.last_pvalues)
        if len(a) or len(b) or split.fleet_alarm:
            break
    assert n_windows >= 5


def test_split_common_flags_fleet_wide_shift_not_models():
    x = _common_factor_values(n=20, steps=1500, seed=1, shift_all_at=900)
    mon = StreamingMonitor(detector_factory=PageHinkley, n_models=20, split_common=True, alpha=0.05,
                           n_ref=300, window=100, horizon=3, calibration=CAL)
    fleet_steps, model_alarms = [], 0
    for t in range(x.shape[1]):
        model_alarms += len(mon.update(x[:, t]))
        if mon.fleet_alarm:
            fleet_steps.append(t)
    assert fleet_steps and 900 <= fleet_steps[0] < 1300
    assert model_alarms <= 2


def test_split_common_requires_every_model_and_survives_save(tmp_path):
    x = _common_factor_values(n=6, steps=900, seed=3)
    ids = list("abcdef")
    kw = dict(procedure="bh_window", alpha=0.3, n_ref=200, window=100, horizon=2, calibration=CAL, seed=4,
              split_common=True)
    mon = StreamingMonitor(detector_factory=PageHinkley, model_ids=ids, **kw)
    with pytest.raises(ValueError):
        mon.update({"a": 1.0})
    straight = StreamingMonitor(detector_factory=PageHinkley, model_ids=ids, **kw)
    first = _feed(straight, x, 0, 450, ids)
    resumed = StreamingMonitor(detector_factory=PageHinkley, model_ids=ids, **kw)
    assert _feed(resumed, x, 0, 450, ids) == first
    resumed.reset_model("c")  # a retrained model re-estimates its scale
    straight2 = StreamingMonitor(detector_factory=PageHinkley, model_ids=ids, **kw)
    _feed(straight2, x, 0, 450, ids)
    straight2.reset_model("c")
    resumed.save(tmp_path / "split.npz")
    loaded = StreamingMonitor.load(tmp_path / "split.npz", detector_factory=PageHinkley)
    out_loaded = _feed(loaded, x, 450, 900, ids)
    assert out_loaded == _feed(straight2, x, 450, 900, ids)
    assert loaded.n_seen == straight2.n_seen and loaded._scale == straight2._scale
