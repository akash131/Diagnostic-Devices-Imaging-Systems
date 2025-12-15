"""Imaging Systems Module.

This module provides implementations for various diagnostic imaging devices:
- Ultrasound transducers and signal processing units
- Portable X-ray systems
- Optical coherence tomography (OCT) devices
- Thermographic cameras for inflammation detection
"""

from .base import ImagingDevice, ImageData, DeviceStatus
from .ultrasound import UltrasoundTransducer, SignalProcessor
from .xray import PortableXRay, XRayDetector
from .oct import OCTDevice, OCTScanner
from .thermographic import ThermographicCamera, InflammationDetector

__all__ = [
    "ImagingDevice",
    "ImageData",
    "DeviceStatus",
    "UltrasoundTransducer",
    "SignalProcessor",
    "PortableXRay",
    "XRayDetector",
    "OCTDevice",
    "OCTScanner",
    "ThermographicCamera",
    "InflammationDetector",
]
