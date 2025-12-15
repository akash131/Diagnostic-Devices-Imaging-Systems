"""Vendor adapters for ultrasound equipment manufacturers."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import numpy as np

from .base import (
    VendorAdapter,
    VendorProtocol,
    ConnectionConfig,
    DeviceCapabilities,
    DataFormat,
    SimulatedProtocol,
)
from ..base import ImageData
from ..dicom.builder import ModalityBuilder


class GEHealthcareUltrasound(VendorAdapter):
    """Adapter for GE Healthcare ultrasound systems."""

    VENDOR_NAME = "GE Healthcare"
    VENDOR_ID = "ge_healthcare_us"
    SUPPORTED_MODELS = [
        "LOGIQ E10",
        "LOGIQ E9",
        "LOGIQ S8",
        "Voluson E10",
        "Voluson E8",
        "Vivid E95",
        "Vivid E90",
        "Venue Go",
        "Vscan",
    ]

    def __init__(self, model: str = "LOGIQ E10"):
        super().__init__(model)
        self._protocol = SimulatedProtocol()
        self._imaging_mode = "b_mode"
        self._preset = "general"

    def get_capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            modalities=["US"],
            supported_formats=[DataFormat.DICOM, DataFormat.PROPRIETARY, DataFormat.RAW],
            max_resolution=(1920, 1080),
            supports_streaming=True,
            supports_remote_control=True,
            supports_calibration=True,
            supports_dicom_export=True,
            max_frame_rate=60.0,
            color_depth=8,
            features=[
                "B-Mode",
                "M-Mode",
                "Color Doppler",
                "Power Doppler",
                "Pulsed Wave Doppler",
                "Contrast Imaging",
                "Elastography",
                "3D/4D Imaging",
                "Tissue Harmonic Imaging",
            ],
        )

    def connect(self, config: ConnectionConfig) -> bool:
        try:
            self._connected = self._protocol.connect(config)
            if self._connected:
                self._emit_event("connected", {"vendor": self.VENDOR_NAME})
            return self._connected
        except Exception as e:
            self._last_error = str(e)
            return False

    def disconnect(self) -> bool:
        self._connected = False
        self._protocol.disconnect()
        self._emit_event("disconnected", {"vendor": self.VENDOR_NAME})
        return True

    def acquire_image(self, params: dict = None) -> ImageData:
        params = params or {}

        # Simulate GE-specific acquisition
        resolution = params.get("resolution", (800, 600))
        depth = params.get("depth", 100)
        gain = params.get("gain", 50)

        # Generate simulated ultrasound image
        image = self._generate_simulated_image(resolution, depth, gain)

        return ImageData(
            data=image,
            timestamp=datetime.now(),
            metadata={
                "vendor": self.VENDOR_NAME,
                "model": self.model,
                "modality": "ultrasound",
                "imaging_mode": self._imaging_mode,
                "depth_mm": depth,
                "gain": gain,
                "preset": self._preset,
                "transducer": params.get("transducer", "C5-1"),
            },
            device_id=f"{self.VENDOR_ID}_{self.model}",
            modality="ultrasound",
        )

    def _generate_simulated_image(
        self,
        resolution: tuple[int, int],
        depth: float,
        gain: float,
    ) -> np.ndarray:
        """Generate simulated ultrasound image with GE-like appearance."""
        height, width = resolution
        image = np.random.rayleigh(scale=20, size=(height, width))

        # Add sector shape mask
        y, x = np.ogrid[:height, :width]
        center_x = width // 2
        angle = np.arctan2(x - center_x, y)
        mask = np.abs(angle) < np.pi / 3  # 60-degree sector
        image *= mask

        # Apply gain
        image *= gain / 50

        # Add some structure
        for _ in range(10):
            cy = np.random.randint(height // 4, 3 * height // 4)
            cx = np.random.randint(width // 4, 3 * width // 4)
            r = np.random.randint(10, 50)
            y_idx, x_idx = np.ogrid[:height, :width]
            dist = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
            image[dist < r] *= 1.5

        return np.clip(image, 0, 255).astype(np.uint8)

    def configure_device(self, settings: dict) -> bool:
        if "imaging_mode" in settings:
            self._imaging_mode = settings["imaging_mode"]
        if "preset" in settings:
            self._preset = settings["preset"]
        return True

    def get_device_status(self) -> dict:
        return {
            "vendor": self.VENDOR_NAME,
            "model": self.model,
            "connected": self._connected,
            "imaging_mode": self._imaging_mode,
            "preset": self._preset,
        }

    def export_to_dicom(self, image_data: ImageData) -> bytes:
        dataset = ModalityBuilder.create_ultrasound_dataset(
            image_data.data,
            patient_name="Anonymous",
            patient_id="000000",
            manufacturer=self.VENDOR_NAME,
            model=self.model,
        )
        # Simplified export
        return str(dataset.to_dict()).encode()

    def import_from_vendor_format(self, data: bytes) -> ImageData:
        # Parse GE proprietary format
        # This is a simplified implementation
        image = np.frombuffer(data, dtype=np.uint8)
        return ImageData(
            data=image,
            timestamp=datetime.now(),
            metadata={"vendor": self.VENDOR_NAME, "format": "proprietary"},
            modality="ultrasound",
        )

    def get_supported_settings(self) -> dict:
        return {
            "depth": {"min": 10, "max": 300, "unit": "mm"},
            "gain": {"min": 0, "max": 100, "unit": "%"},
            "frequency": {"min": 1.0, "max": 15.0, "unit": "MHz"},
            "focus": {"min": 10, "max": 200, "unit": "mm"},
            "dynamic_range": {"min": 30, "max": 90, "unit": "dB"},
            "imaging_modes": ["b_mode", "m_mode", "color_doppler", "power_doppler"],
            "presets": ["general", "cardiac", "vascular", "obstetric", "musculoskeletal"],
        }


class PhilipsUltrasound(VendorAdapter):
    """Adapter for Philips ultrasound systems."""

    VENDOR_NAME = "Philips Healthcare"
    VENDOR_ID = "philips_us"
    SUPPORTED_MODELS = [
        "EPIQ Elite",
        "EPIQ 7",
        "EPIQ 5",
        "Affiniti 70",
        "Affiniti 50",
        "Lumify",
        "CX50",
    ]

    def __init__(self, model: str = "EPIQ 7"):
        super().__init__(model)
        self._protocol = SimulatedProtocol()

    def get_capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            modalities=["US"],
            supported_formats=[DataFormat.DICOM, DataFormat.PROPRIETARY],
            max_resolution=(1920, 1080),
            supports_streaming=True,
            supports_remote_control=True,
            max_frame_rate=50.0,
            features=[
                "PureWave Technology",
                "xMATRIX",
                "XRES",
                "SonoCT",
                "3D/4D",
                "Elastography",
                "Contrast Imaging",
            ],
        )

    def connect(self, config: ConnectionConfig) -> bool:
        self._connected = self._protocol.connect(config)
        return self._connected

    def disconnect(self) -> bool:
        self._connected = False
        return self._protocol.disconnect()

    def acquire_image(self, params: dict = None) -> ImageData:
        params = params or {}
        resolution = params.get("resolution", (800, 600))

        image = np.random.rayleigh(scale=25, size=resolution).astype(np.uint8)

        return ImageData(
            data=image,
            timestamp=datetime.now(),
            metadata={
                "vendor": self.VENDOR_NAME,
                "model": self.model,
                "modality": "ultrasound",
            },
            modality="ultrasound",
        )

    def configure_device(self, settings: dict) -> bool:
        return True

    def get_device_status(self) -> dict:
        return {"vendor": self.VENDOR_NAME, "model": self.model, "connected": self._connected}

    def export_to_dicom(self, image_data: ImageData) -> bytes:
        dataset = ModalityBuilder.create_ultrasound_dataset(
            image_data.data, "Anonymous", "000000",
            manufacturer=self.VENDOR_NAME, model=self.model,
        )
        return str(dataset.to_dict()).encode()

    def import_from_vendor_format(self, data: bytes) -> ImageData:
        return ImageData(
            data=np.frombuffer(data, dtype=np.uint8),
            timestamp=datetime.now(),
            metadata={"vendor": self.VENDOR_NAME},
            modality="ultrasound",
        )


class SiemensUltrasound(VendorAdapter):
    """Adapter for Siemens Healthineers ultrasound systems."""

    VENDOR_NAME = "Siemens Healthineers"
    VENDOR_ID = "siemens_us"
    SUPPORTED_MODELS = [
        "ACUSON Sequoia",
        "ACUSON Redwood",
        "ACUSON Juniper",
        "ACUSON P500",
        "ACUSON NX3",
    ]

    def __init__(self, model: str = "ACUSON Sequoia"):
        super().__init__(model)
        self._protocol = SimulatedProtocol()

    def get_capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            modalities=["US"],
            supported_formats=[DataFormat.DICOM, DataFormat.PROPRIETARY],
            max_resolution=(1920, 1080),
            supports_streaming=True,
            max_frame_rate=55.0,
            features=["BioAcoustic", "eSie Touch", "Virtual Touch", "ARFI"],
        )

    def connect(self, config: ConnectionConfig) -> bool:
        self._connected = self._protocol.connect(config)
        return self._connected

    def disconnect(self) -> bool:
        self._connected = False
        return True

    def acquire_image(self, params: dict = None) -> ImageData:
        return ImageData(
            data=np.random.rayleigh(scale=22, size=(600, 800)).astype(np.uint8),
            timestamp=datetime.now(),
            metadata={"vendor": self.VENDOR_NAME, "model": self.model},
            modality="ultrasound",
        )

    def configure_device(self, settings: dict) -> bool:
        return True

    def get_device_status(self) -> dict:
        return {"vendor": self.VENDOR_NAME, "connected": self._connected}

    def export_to_dicom(self, image_data: ImageData) -> bytes:
        return b""

    def import_from_vendor_format(self, data: bytes) -> ImageData:
        return ImageData(data=np.array([]), timestamp=datetime.now(), modality="ultrasound")


class CanonUltrasound(VendorAdapter):
    """Adapter for Canon Medical (formerly Toshiba) ultrasound systems."""

    VENDOR_NAME = "Canon Medical Systems"
    VENDOR_ID = "canon_us"
    SUPPORTED_MODELS = ["Aplio i900", "Aplio i800", "Aplio i700", "Aplio a550", "Xario 200"]

    def __init__(self, model: str = "Aplio i800"):
        super().__init__(model)
        self._protocol = SimulatedProtocol()

    def get_capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            modalities=["US"],
            supported_formats=[DataFormat.DICOM],
            max_resolution=(1920, 1080),
            supports_streaming=True,
            max_frame_rate=50.0,
            features=["iBeam", "Superb Micro-vascular Imaging", "Shear Wave Elastography"],
        )

    def connect(self, config: ConnectionConfig) -> bool:
        self._connected = self._protocol.connect(config)
        return self._connected

    def disconnect(self) -> bool:
        self._connected = False
        return True

    def acquire_image(self, params: dict = None) -> ImageData:
        return ImageData(
            data=np.random.rayleigh(scale=20, size=(600, 800)).astype(np.uint8),
            timestamp=datetime.now(),
            metadata={"vendor": self.VENDOR_NAME},
            modality="ultrasound",
        )

    def configure_device(self, settings: dict) -> bool:
        return True

    def get_device_status(self) -> dict:
        return {"vendor": self.VENDOR_NAME, "connected": self._connected}

    def export_to_dicom(self, image_data: ImageData) -> bytes:
        return b""

    def import_from_vendor_format(self, data: bytes) -> ImageData:
        return ImageData(data=np.array([]), timestamp=datetime.now(), modality="ultrasound")


class SamsungUltrasound(VendorAdapter):
    """Adapter for Samsung ultrasound systems."""

    VENDOR_NAME = "Samsung Medison"
    VENDOR_ID = "samsung_us"
    SUPPORTED_MODELS = ["RS85 Prestige", "RS80A", "HS70A", "HM70A", "PT60A"]

    def __init__(self, model: str = "RS85 Prestige"):
        super().__init__(model)
        self._protocol = SimulatedProtocol()

    def get_capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            modalities=["US"],
            supported_formats=[DataFormat.DICOM],
            max_resolution=(1920, 1080),
            supports_streaming=True,
            max_frame_rate=45.0,
            features=["CrystalLive", "S-Detect", "E-Breast", "RealisticVue"],
        )

    def connect(self, config: ConnectionConfig) -> bool:
        self._connected = self._protocol.connect(config)
        return self._connected

    def disconnect(self) -> bool:
        self._connected = False
        return True

    def acquire_image(self, params: dict = None) -> ImageData:
        return ImageData(
            data=np.random.rayleigh(scale=23, size=(600, 800)).astype(np.uint8),
            timestamp=datetime.now(),
            metadata={"vendor": self.VENDOR_NAME},
            modality="ultrasound",
        )

    def configure_device(self, settings: dict) -> bool:
        return True

    def get_device_status(self) -> dict:
        return {"vendor": self.VENDOR_NAME, "connected": self._connected}

    def export_to_dicom(self, image_data: ImageData) -> bytes:
        return b""

    def import_from_vendor_format(self, data: bytes) -> ImageData:
        return ImageData(data=np.array([]), timestamp=datetime.now(), modality="ultrasound")
