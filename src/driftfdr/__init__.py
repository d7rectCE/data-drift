"""Online FDR control on top of practical drift detectors for multi-model monitoring."""

from .calibration import CalibrationConfig, NullDistribution, calibrate, calibrate_many
from .detectors import ADWIN, DDM, Detector, KSSliding, KSWindow, MeanShift, PageHinkley, default_detectors
from .metrics import summarize
from .monitor import MonitorConfig, MonitorResult, run_monitor
from .online_fdr import (
    LOND,
    SAFFRON,
    AlphaInvesting,
    BatchBH,
    BHWindow,
    BonferroniWindow,
    EBHWindow,
    LORDpp,
    RawThreshold,
    StoreyBHWindow,
    Uncorrected,
    make_procedure,
)
from .streaming import CalibratedDetector, StreamingMonitor, from_river
from .streams import Scenario, ScenarioConfig, SupervisedConfig, make_scenario, make_supervised_scenario

__all__ = [
    "ADWIN",
    "DDM",
    "LOND",
    "SAFFRON",
    "AlphaInvesting",
    "BHWindow",
    "BatchBH",
    "CalibratedDetector",
    "BonferroniWindow",
    "CalibrationConfig",
    "EBHWindow",
    "Detector",
    "KSSliding",
    "KSWindow",
    "LORDpp",
    "MeanShift",
    "MonitorConfig",
    "MonitorResult",
    "NullDistribution",
    "PageHinkley",
    "RawThreshold",
    "Scenario",
    "ScenarioConfig",
    "StoreyBHWindow",
    "StreamingMonitor",
    "SupervisedConfig",
    "Uncorrected",
    "calibrate",
    "calibrate_many",
    "default_detectors",
    "from_river",
    "make_procedure",
    "make_scenario",
    "make_supervised_scenario",
    "run_monitor",
    "summarize",
]
