"""Online FDR control on top of practical drift detectors for multi-model monitoring."""

from .calibration import CalibrationConfig, NullDistribution, calibrate, calibrate_many
from .detectors import ADWIN, DDM, Detector, KSWindow, PageHinkley, default_detectors
from .metrics import summarize
from .monitor import MonitorConfig, MonitorResult, run_monitor
from .online_fdr import (
    LOND,
    SAFFRON,
    AlphaInvesting,
    BHWindow,
    BonferroniWindow,
    LORDpp,
    RawThreshold,
    Uncorrected,
    make_procedure,
)
from .streams import Scenario, ScenarioConfig, make_scenario

__all__ = [
    "ADWIN",
    "DDM",
    "LOND",
    "SAFFRON",
    "AlphaInvesting",
    "BHWindow",
    "BonferroniWindow",
    "CalibrationConfig",
    "Detector",
    "KSWindow",
    "LORDpp",
    "MonitorConfig",
    "MonitorResult",
    "NullDistribution",
    "PageHinkley",
    "RawThreshold",
    "Scenario",
    "ScenarioConfig",
    "Uncorrected",
    "calibrate",
    "calibrate_many",
    "default_detectors",
    "make_procedure",
    "make_scenario",
    "run_monitor",
    "summarize",
]
