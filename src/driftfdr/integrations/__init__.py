"""Optional integrations: where the monitor's decisions go, and where its signal comes from.

* ``PrometheusExporter`` publishes p-values, alarms and fleet alarms as Prometheus
  metrics (``pip install driftfdr[prometheus]``), so alerting, dashboards and history
  come from the stack a team already runs.
* ``MLflowReporter`` logs p-values to an MLflow run and tags the registered model
  version of an alarmed model (``pip install driftfdr[mlflow]``), so a retraining
  pipeline can pick it up.
* ``cbpe_estimated_error`` turns NannyML's label-free performance estimate (CBPE)
  into an error series to monitor while labels are delayed
  (``pip install driftfdr[nannyml]``).

Each integration imports its package only when used, so the core library keeps its
small set of dependencies.
"""

from .mlflow import MLflowReporter
from .nannyml import cbpe_estimated_error
from .prometheus import PrometheusExporter

__all__ = ["MLflowReporter", "PrometheusExporter", "cbpe_estimated_error"]
