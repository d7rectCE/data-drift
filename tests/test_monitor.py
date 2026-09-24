import numpy as np
import pytest

from driftfdr import (
    CalibrationConfig,
    MonitorConfig,
    PageHinkley,
    RawThreshold,
    ScenarioConfig,
    Uncorrected,
    make_procedure,
    make_scenario,
    run_monitor,
    summarize,
)

CONFIG = MonitorConfig(n_ref=200, window=100, calibration=CalibrationConfig(n_boot=100))


@pytest.fixture(scope="module")
def scenario():
    return make_scenario(
        ScenarioConfig(n_streams=20, n_steps=2000, phi=0.3, drift_fraction=0.25, magnitude=1.5),
        seed=0,
    )


def test_scenario_ground_truth(scenario):
    drifting = scenario.drifting
    assert drifting.size == 5
    k = drifting[0]
    tau = scenario.change_start[k]
    assert scenario.is_null([k], 0, tau)[0]
    assert not scenario.is_null([k], 0, tau + 1)[0]
    assert scenario.is_null([k], scenario.change_end[k], scenario.n_steps)[0]


def test_retrained_stream_pauses_for_reference(scenario):
    result = run_monitor(scenario, PageHinkley(), Uncorrected(0.2), CONFIG, seed=0)
    tests = result.tests
    for _, alarm in tests[tests["rejected"]].iterrows():
        later = tests[(tests["stream"] == alarm["stream"]) & (tests["t_end"] > alarm["t_end"])]
        if not later.empty:
            assert later["t_end"].min() >= alarm["t_end"] + CONFIG.n_ref + CONFIG.window
            assert later["ref_start"].iloc[0] == alarm["t_end"]


def test_strong_drifts_are_detected(scenario):
    result = run_monitor(scenario, PageHinkley(), make_procedure("LORD++", 0.1), CONFIG, seed=0)
    s = summarize(result)
    assert s["n_drifts"] == 5
    assert s["detected"] >= 4
    assert s["mean_delay"] < 400


def test_cache_is_reused_across_procedures(scenario):
    cache = {}
    run_monitor(scenario, PageHinkley(), make_procedure("bh_window", 0.1), CONFIG, cache=cache)
    n_before = len(cache)
    run_monitor(scenario, PageHinkley(), make_procedure("SAFFRON", 0.1), CONFIG, cache=cache)
    assert n_before >= scenario.n_streams
    assert len(cache) <= n_before + scenario.drifting.size + 5


def test_raw_threshold_needs_no_calibration(scenario):
    result = run_monitor(scenario, PageHinkley(), RawThreshold(50.0), CONFIG)
    assert result.tests["pvalue"].isna().all()
    assert np.isfinite(summarize(result)["fdp"])
