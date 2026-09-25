"""Log a ``StreamingMonitor``'s p-values to MLflow and tag alarmed model versions."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone


def _key(model_id) -> str:
    """MLflow metric keys allow letters, digits, ``_ - . /`` and spaces."""
    return re.sub(r"[^\w\-./ ]", "_", str(model_id))


class MLflowReporter:
    """Wraps a ``StreamingMonitor``: p-values go to an MLflow run, alarms to the model registry.

    * With ``run_id``, every p-value is logged as the metric ``pvalue.<model>`` and every
      alarm as ``alarm.<model>`` = 1 (and ``fleet_alarm`` for the common component), with
      the monitor's step as the MLflow step.
    * ``model_versions`` maps a monitored model to its registered model ``(name, version)``;
      when it alarms, that version gets the tag ``tag`` (default ``driftfdr_retrain``) with
      the time and p-value, which a retraining pipeline can query. After retraining, point
      the model to its new version with ``set_version``.

    Use ``update`` instead of ``monitor.update``. Requires ``mlflow``; pass ``client`` to
    reuse an ``MlflowClient``.
    """

    MAX_BATCH = 1000  # MLflow's limit of metrics per log_batch call

    def __init__(self, monitor, run_id: str | None = None, model_versions=None, tag: str = "driftfdr_retrain",
                 client=None, tracking_uri: str | None = None):
        if client is None:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri)
        self.monitor, self.client, self.run_id, self.tag = monitor, client, run_id, tag
        self.model_versions = {m: (name, str(version)) for m, (name, version) in dict(model_versions or {}).items()}
        self.step = 0

    def set_version(self, model_id, name: str, version) -> None:
        """Registered model version that stands behind ``model_id`` from now on."""
        self.model_versions[model_id] = (name, str(version))

    def update(self, observations):
        """``monitor.update(observations)``, then report; returns the alarmed models."""
        alarmed = self.monitor.update(observations)
        self.record(alarmed)
        return alarmed

    def record(self, alarmed) -> None:
        """Report the outcome of a ``monitor.update`` made elsewhere that returned ``alarmed``."""
        self.step += 1
        if self.run_id is not None:
            from mlflow.entities import Metric

            ts = int(time.time() * 1000)
            metrics = [Metric(f"pvalue.{_key(m)}", float(p), ts, self.step)
                       for m, p in self.monitor.last_pvalues.items() if m != self.monitor.FLEET]
            metrics += [Metric(f"alarm.{_key(m)}", 1.0, ts, self.step) for m in alarmed]
            if self.monitor.fleet_alarm:
                metrics.append(Metric("fleet_alarm", 1.0, ts, self.step))
            for i in range(0, len(metrics), self.MAX_BATCH):
                self.client.log_batch(self.run_id, metrics=metrics[i : i + self.MAX_BATCH])
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for m in alarmed:
            if m in self.model_versions:
                name, version = self.model_versions[m]
                p = self.monitor.last_pvalues.get(m)
                value = stamp if p is None else f"{stamp} p={p:.3g}"
                self.client.set_model_version_tag(name, version, self.tag, value)
