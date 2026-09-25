import sys
import types
from collections import namedtuple

import numpy as np
import pytest

from driftfdr import CalibrationConfig, PageHinkley, ScenarioConfig, StreamingMonitor, make_scenario

CAL = CalibrationConfig(n_boot=100, method="sieve")


def _drifting_monitor():
    sc = make_scenario(ScenarioConfig(n_streams=4, n_steps=900, phi=0.3, drift_fraction=0.5, magnitude=2.5,
                                      fixed_onset=450), seed=11)
    mon = StreamingMonitor(detector_factory=PageHinkley, model_ids=["a", "b", "c", "d"], n_ref=200, window=100,
                           horizon=3, calibration=CAL)
    return sc, mon


def test_prometheus_exporter_counts_alarms_and_pvalues():
    prometheus_client = pytest.importorskip("prometheus_client")
    from driftfdr.integrations import PrometheusExporter

    sc, mon = _drifting_monitor()
    registry = prometheus_client.CollectorRegistry()
    exporter = PrometheusExporter(mon, registry=registry)
    alarms = []
    for t in range(sc.n_steps):
        alarms += exporter.update({m: sc.values[i, t] for i, m in enumerate("abcd")})
    assert alarms
    get = registry.get_sample_value
    assert sum(get("driftfdr_alarms_total", {"model": m}) or 0 for m in "abcd") == len(alarms)
    assert 0 < get("driftfdr_pvalue", {"model": "a"}) <= 1
    assert get("driftfdr_models") == 4 and get("driftfdr_calibrated", {"model": "a"}) in (0.0, 1.0)
    mon.remove_model("d")
    exporter.forget("d")
    assert get("driftfdr_pvalue", {"model": "d"}) is None


def test_mlflow_reporter_logs_pvalues_and_tags_alarmed_versions(monkeypatch):
    mlflow = types.ModuleType("mlflow")
    entities = types.ModuleType("mlflow.entities")
    entities.Metric = namedtuple("Metric", "key value timestamp step")
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.entities", entities)
    from driftfdr.integrations import MLflowReporter

    class Client:
        def __init__(self):
            self.metrics, self.tags = [], []

        def log_batch(self, run_id, metrics):
            self.metrics += metrics

        def set_model_version_tag(self, name, version, key, value):
            self.tags.append((name, version, key, value))

    sc, mon = _drifting_monitor()
    client = Client()
    reporter = MLflowReporter(mon, run_id="run1", client=client,
                              model_versions={m: (f"model-{m}", 1) for m in "abcd"})
    alarms = []
    for t in range(sc.n_steps):
        alarms += reporter.update({m: sc.values[i, t] for i, m in enumerate("abcd")})
    assert alarms
    assert {k.key.split(".")[0] for k in client.metrics} >= {"pvalue", "alarm"}
    assert sorted(t[0] for t in client.tags) == sorted(f"model-{m}" for m in alarms)
    assert all(t[1] == "1" and t[2] == "driftfdr_retrain" and "p=" in t[3] for t in client.tags)


def test_cbpe_estimated_error_tracks_true_error():
    pytest.importorskip("nannyml")
    import pandas as pd

    from driftfdr.integrations import cbpe_estimated_error

    rng = np.random.default_rng(0)

    def frame(n, shift):
        x = rng.normal(shift, 1, n)
        proba = 1 / (1 + np.exp(-2 * x))
        y = (rng.random(n) < proba).astype(int)
        return pd.DataFrame({"y_pred_proba": proba, "y_pred": (proba > 0.5).astype(int), "y_true": y})

    est = cbpe_estimated_error(frame(5000, 0.0), pd.concat([frame(2000, 0.0), frame(2000, 1.5)]), chunk_size=500)
    assert est.shape == (8,) and est[:4].mean() > est[4:].mean() + 0.05  # confident inputs -> fewer errors
