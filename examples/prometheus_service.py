"""A drift monitor as a small service: read errors, decide, expose metrics to Prometheus.

Run it, point Prometheus at http://localhost:8000/metrics (examples/prometheus/prometheus.yml)
and load the alert rules (examples/prometheus/alerts.yml); Alertmanager and Grafana then do
the notifying and plotting. The state is saved to a file on exit and every ``--save-every``
steps, so a restart resumes where it stopped.

The stand-in data source is a synthetic fleet; replace ``error_stream`` with a reader of
your own (a Kafka consumer, a query of the latest labelled predictions, ...) that yields
``{model_id: error}`` for each step.

Usage: python examples/prometheus_service.py [--port 8000] [--state monitor.npz] [--steps 6000]
Requires: pip install driftfdr[prometheus]
"""

import argparse
import os
import signal
import time

from driftfdr import ECUSUM, CalibrationConfig, ScenarioConfig, StreamingMonitor, make_scenario
from driftfdr.integrations import PrometheusExporter

MODELS = [f"model-{i:02d}" for i in range(20)]


def error_stream(n_steps):
    """Stand-in for live data: per-step errors of 20 models, a few of which drift."""
    scenario = make_scenario(ScenarioConfig(n_streams=len(MODELS), n_steps=n_steps, drift_fraction=0.15), seed=0)
    for t in range(n_steps):
        yield {m: float(scenario.values[i, t]) for i, m in enumerate(MODELS)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--state", default="monitor.npz")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--interval", type=float, default=0.01, help="seconds between steps (stand-in data only)")
    parser.add_argument("--save-every", type=int, default=500)
    args = parser.parse_args()

    if os.path.exists(args.state):
        monitor = StreamingMonitor.load(args.state, detector_factory=ECUSUM)
        print(f"resumed from {args.state}")
    else:
        monitor = StreamingMonitor(detector_factory=ECUSUM, model_ids=MODELS, sequential=True, alpha=0.05,
                                   calibration=CalibrationConfig(n_boot=500))
    exporter = PrometheusExporter(monitor)
    exporter.serve(args.port)
    print(f"metrics at http://localhost:{args.port}/metrics")

    def save_and_exit(*_):
        monitor.save(args.state)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, save_and_exit)
    try:
        for step, errors in enumerate(error_stream(args.steps), 1):
            for model_id in exporter.update(errors):
                print(f"step {step}: retrain {model_id}")  # or trigger the retraining pipeline here
            if step % args.save_every == 0:
                monitor.save(args.state)
            time.sleep(args.interval)
    finally:
        monitor.save(args.state)


if __name__ == "__main__":
    main()
