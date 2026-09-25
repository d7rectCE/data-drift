"""Publish a ``StreamingMonitor``'s state as Prometheus metrics."""

from __future__ import annotations

import time


class PrometheusExporter:
    """Wraps a ``StreamingMonitor`` and exposes its decisions as Prometheus metrics.

    Metrics (``prefix`` defaults to ``driftfdr``), labelled by ``model`` where it applies:

    * ``<prefix>_pvalue`` — the latest p-value of each model (set when one becomes ready);
    * ``<prefix>_alarms_total`` — alarms (retrain decisions) per model;
    * ``<prefix>_last_alarm_timestamp_seconds`` — when each model last alarmed;
    * ``<prefix>_calibrated`` — 1 once a model's reference is complete, 0 while collecting;
    * ``<prefix>_fleet_alarms_total`` — alarms of the common component (``split_common``);
    * ``<prefix>_models`` — number of monitored models.

    Use ``update`` instead of ``monitor.update``; ``serve(port)`` starts the HTTP endpoint
    Prometheus scrapes. Alert on ``increase(driftfdr_alarms_total[1h]) > 0``, see
    ``examples/prometheus/``. Requires ``prometheus_client``.
    """

    def __init__(self, monitor, registry=None, prefix: str = "driftfdr"):
        from prometheus_client import REGISTRY, Counter, Gauge

        self.monitor = monitor
        self.registry = REGISTRY if registry is None else registry
        kw = dict(registry=self.registry)
        self.pvalue = Gauge(f"{prefix}_pvalue", "Latest p-value of the model's drift test", ["model"], **kw)
        self.alarms = Counter(f"{prefix}_alarms", "Alarms (retrain decisions) of the model", ["model"], **kw)
        self.last_alarm = Gauge(f"{prefix}_last_alarm_timestamp_seconds", "Time of the model's last alarm",
                                ["model"], **kw)
        self.calibrated = Gauge(f"{prefix}_calibrated", "1 once the model's reference is complete", ["model"], **kw)
        self.fleet_alarms = Counter(f"{prefix}_fleet_alarms", "Alarms of the fleet's common component", **kw)
        self.n_models = Gauge(f"{prefix}_models", "Number of monitored models", **kw)

    def update(self, observations):
        """``monitor.update(observations)``, then refresh the metrics; returns the alarmed models."""
        alarmed = self.monitor.update(observations)
        self.record(alarmed)
        return alarmed

    def record(self, alarmed) -> None:
        """Refresh the metrics after a ``monitor.update`` made elsewhere that returned ``alarmed``."""
        now = time.time()
        for model_id, p in self.monitor.last_pvalues.items():
            if model_id != self.monitor.FLEET:
                self.pvalue.labels(model=str(model_id)).set(p)
        for model_id in alarmed:
            self.alarms.labels(model=str(model_id)).inc()
            self.last_alarm.labels(model=str(model_id)).set(now)
        if self.monitor.fleet_alarm:
            self.fleet_alarms.inc()
        for model_id, det in self.monitor.models.items():
            self.calibrated.labels(model=str(model_id)).set(1.0 if det.calibrated else 0.0)
        self.n_models.set(len(self.monitor.models))

    def forget(self, model_id) -> None:
        """Drop a removed model's series (call after ``monitor.remove_model``)."""
        for metric in (self.pvalue, self.alarms, self.last_alarm, self.calibrated):
            try:
                metric.remove(str(model_id))
            except KeyError:
                pass

    def serve(self, port: int = 8000, addr: str = "0.0.0.0"):
        """Start the HTTP endpoint that Prometheus scrapes, in a background thread."""
        from prometheus_client import start_http_server

        return start_http_server(port, addr=addr, registry=self.registry)
