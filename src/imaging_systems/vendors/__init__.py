"""Vendor-specific adapters and protocol implementations."""

from .base import VendorAdapter, VendorProtocol, ConnectionConfig
from .ultrasound_vendors import (
    GEHealthcareUltrasound,
    PhilipsUltrasound,
    SiemensUltrasound,
    CanonUltrasound,
    SamsungUltrasound,
)
from .xray_vendors import (
    GEHealthcareXRay,
    PhilipsXRay,
    SiemensXRay,
    CareStreamXRay,
    FujifilmXRay,
)
from .oct_vendors import (
    ZeissOCT,
    HeidelbergOCT,
    TopconOCT,
    OptovueOCT,
)
from .thermal_vendors import (
    FLIRThermal,
    InfraTecThermal,
    TestoThermal,
)
from .registry import VendorRegistry, get_vendor_adapter

__all__ = [
    "VendorAdapter",
    "VendorProtocol",
    "ConnectionConfig",
    "GEHealthcareUltrasound",
    "PhilipsUltrasound",
    "SiemensUltrasound",
    "CanonUltrasound",
    "SamsungUltrasound",
    "GEHealthcareXRay",
    "PhilipsXRay",
    "SiemensXRay",
    "CareStreamXRay",
    "FujifilmXRay",
    "ZeissOCT",
    "HeidelbergOCT",
    "TopconOCT",
    "OptovueOCT",
    "FLIRThermal",
    "InfraTecThermal",
    "TestoThermal",
    "VendorRegistry",
    "get_vendor_adapter",
]
