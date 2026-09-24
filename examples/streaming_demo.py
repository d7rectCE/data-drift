"""Monitoring 50 models online: one observation per model per step, alarms as they happen."""

from river import drift

from driftfdr import CalibrationConfig, ScenarioConfig, StreamingMonitor, from_river, make_scenario

# stand-in for live data: per-step loss of 50 models, 10% of which drift
scenario = make_scenario(ScenarioConfig(n_streams=50, n_steps=4000, drift_fraction=0.1), seed=0)

monitor = StreamingMonitor(
    n_models=50,
    detector_factory=lambda: from_river(drift.PageHinkley(mode="up")),
    procedure="bh_window",  # or "bonferroni", "LORD++", ...
    alpha=0.05,
    calibration=CalibrationConfig(n_boot=500, method="sieve"),
)

for t in range(scenario.n_steps):
    for k in monitor.update(scenario.values[:, t]):
        truth = "дрейф" if scenario.change_start[k] < t else "ложная тревога"
        print(f"шаг {t + 1}: переобучить модель {k} ({truth})")
