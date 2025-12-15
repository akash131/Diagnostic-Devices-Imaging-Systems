"""Thermographic cameras for inflammation detection module."""

from .camera import ThermographicCamera, ThermalSensor, SensorType
from .inflammation_detector import (
    InflammationDetector,
    InflammationRegion,
    AnalysisResult,
)

__all__ = [
    "ThermographicCamera",
    "ThermalSensor",
    "SensorType",
    "InflammationDetector",
    "InflammationRegion",
    "AnalysisResult",
]
