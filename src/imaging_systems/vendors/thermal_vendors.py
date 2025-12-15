"""Vendor adapters for thermal imaging equipment manufacturers."""

from datetime import datetime
import numpy as np

from .base import (
    VendorAdapter,
    ConnectionConfig,
    DeviceCapabilities,
    DataFormat,
    SimulatedProtocol,
)
from ..base import ImageData
from ..dicom.builder import ModalityBuilder


class FLIRThermal(VendorAdapter):
    """Adapter for FLIR thermal imaging systems."""

    VENDOR_NAME = "FLIR Systems"
    VENDOR_ID = "flir_thermal"
    SUPPORTED_MODELS = [
        "T1020",
        "T865",
        "T540",
        "E96",
        "E54",
        "ONE Pro",
        "FLIR C5",
    ]

    def __init__(self, model: str = "T865"):
        super().__init__(model)
        self._protocol = SimulatedProtocol()
        self._emissivity = 0.98
        self._range = "auto"
        self._palette = "iron"

    def get_capabilities(self) -> DeviceCapabilities:
        model_resolutions = {
            "T1020": (1024, 768),
            "T865": (640, 480),
            "T540": (464, 348),
            "E96": (640, 480),
            "E54": (320, 240),
            "ONE Pro": (160, 120),
            "FLIR C5": (160, 120),
        }

        return DeviceCapabilities(
            modalities=["TG"],  # Thermography
            supported_formats=[DataFormat.DICOM, DataFormat.PROPRIETARY, DataFormat.JPEG],
            max_resolution=model_resolutions.get(self.model, (640, 480)),
            supports_streaming=True,
            supports_remote_control=True,
            supports_calibration=True,
            supports_dicom_export=True,
            max_frame_rate=30.0,
            color_depth=14,
            features=[
                "MSX Enhancement",
                "1-Touch Level/Span",
                "Laser Pointer",
                "FLIR Tools",
                "UltraMax",
                "Screening Mode",
                "Radiometric Streaming",
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

        emissivity = params.get("emissivity", self._emissivity)
        temp_range = params.get("range", self._range)

        resolution = self.get_capabilities().max_resolution
        image, min_temp, max_temp = self._generate_thermal_image(resolution, emissivity)

        return ImageData(
            data=image,
            timestamp=datetime.now(),
            metadata={
                "vendor": self.VENDOR_NAME,
                "model": self.model,
                "modality": "thermal",
                "emissivity": emissivity,
                "min_temp_c": min_temp,
                "max_temp_c": max_temp,
                "range_mode": temp_range,
                "palette": self._palette,
                "spectral_range_um": (7.5, 14.0),
                "netd_mk": 30,
            },
            device_id=f"{self.VENDOR_ID}_{self.model}",
            modality="thermal",
        )

    def _generate_thermal_image(
        self,
        resolution: tuple[int, int],
        emissivity: float,
    ) -> tuple[np.ndarray, float, float]:
        """Generate simulated FLIR thermal image."""
        height, width = resolution

        # Base body temperature distribution
        base_temp = 33.0  # Surface skin temp

        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2

        # Create body-shaped heat pattern
        dist = np.sqrt(((x - center_x) / (width / 2)) ** 2 + ((y - center_y) / (height / 2)) ** 2)

        # Temperature decreases toward edges
        temp_image = base_temp + 3 * (1 - dist)

        # Add anatomical features
        # Face region (warmer)
        face_region = ((y - center_y * 0.5) ** 2 / (height * 0.15) ** 2 +
                       (x - center_x) ** 2 / (width * 0.12) ** 2) < 1
        temp_image[face_region] += 1.5

        # Eye region (cooler)
        left_eye = ((y - center_y * 0.4) ** 2 / 200 + (x - center_x - 30) ** 2 / 400) < 1
        right_eye = ((y - center_y * 0.4) ** 2 / 200 + (x - center_x + 30) ** 2 / 400) < 1
        temp_image[left_eye] -= 0.5
        temp_image[right_eye] -= 0.5

        # Apply emissivity
        temp_image *= emissivity ** 0.25

        # Add sensor noise
        noise = np.random.normal(0, 0.03, temp_image.shape)
        temp_image += noise

        min_temp = float(temp_image.min())
        max_temp = float(temp_image.max())

        return temp_image.astype(np.float32), min_temp, max_temp

    def configure_device(self, settings: dict) -> bool:
        if "emissivity" in settings:
            self._emissivity = settings["emissivity"]
        if "range" in settings:
            self._range = settings["range"]
        if "palette" in settings:
            self._palette = settings["palette"]
        return True

    def get_device_status(self) -> dict:
        return {
            "vendor": self.VENDOR_NAME,
            "model": self.model,
            "connected": self._connected,
            "emissivity": self._emissivity,
            "range": self._range,
            "palette": self._palette,
        }

    def export_to_dicom(self, image_data: ImageData) -> bytes:
        # Convert temperature to displayable format
        temp_data = image_data.data
        display_data = ((temp_data - temp_data.min()) / (temp_data.max() - temp_data.min()) * 255)
        display_data = display_data.astype(np.uint8)

        dataset = ModalityBuilder.create_thermal_dataset(
            display_data,
            "Anonymous",
            "000000",
            min_temp=image_data.metadata.get("min_temp_c", 20),
            max_temp=image_data.metadata.get("max_temp_c", 40),
            emissivity=image_data.metadata.get("emissivity", 0.98),
            manufacturer=self.VENDOR_NAME,
            model=self.model,
        )
        return str(dataset.to_dict()).encode()

    def import_from_vendor_format(self, data: bytes) -> ImageData:
        # Parse FLIR radiometric JPEG format
        # Simplified implementation
        return ImageData(
            data=np.frombuffer(data, dtype=np.float32),
            timestamp=datetime.now(),
            metadata={"vendor": self.VENDOR_NAME, "format": "flir_rjpeg"},
            modality="thermal",
        )

    def get_supported_settings(self) -> dict:
        return {
            "emissivity": {"min": 0.01, "max": 1.0, "default": 0.98},
            "distance": {"min": 0.1, "max": 100.0, "unit": "m"},
            "ambient_temp": {"min": -40, "max": 120, "unit": "C"},
            "range_modes": ["auto", "-40_120", "0_650", "300_1500"],
            "palettes": ["iron", "rainbow", "gray", "arctic", "lava"],
        }


class InfraTecThermal(VendorAdapter):
    """Adapter for InfraTec thermal imaging systems."""

    VENDOR_NAME = "InfraTec"
    VENDOR_ID = "infratec_thermal"
    SUPPORTED_MODELS = ["ImageIR 9400", "VarioCAM HD", "PIR uc 180"]

    def __init__(self, model: str = "VarioCAM HD"):
        super().__init__(model)
        self._protocol = SimulatedProtocol()

    def get_capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            modalities=["TG"],
            supported_formats=[DataFormat.DICOM, DataFormat.PROPRIETARY],
            max_resolution=(1024, 768),
            supports_streaming=True,
            max_frame_rate=60.0,
            color_depth=16,
            features=[
                "Lock-in Thermography",
                "Pulse Thermography",
                "Active Thermography",
            ],
        )

    def connect(self, config: ConnectionConfig) -> bool:
        self._connected = self._protocol.connect(config)
        return self._connected

    def disconnect(self) -> bool:
        self._connected = False
        return True

    def acquire_image(self, params: dict = None) -> ImageData:
        return ImageData(
            data=np.random.normal(33, 2, (768, 1024)).astype(np.float32),
            timestamp=datetime.now(),
            metadata={"vendor": self.VENDOR_NAME},
            modality="thermal",
        )

    def configure_device(self, settings: dict) -> bool:
        return True

    def get_device_status(self) -> dict:
        return {"vendor": self.VENDOR_NAME, "connected": self._connected}

    def export_to_dicom(self, image_data: ImageData) -> bytes:
        return b""

    def import_from_vendor_format(self, data: bytes) -> ImageData:
        return ImageData(data=np.array([]), timestamp=datetime.now(), modality="thermal")


class TestoThermal(VendorAdapter):
    """Adapter for Testo thermal imaging systems."""

    VENDOR_NAME = "Testo"
    VENDOR_ID = "testo_thermal"
    SUPPORTED_MODELS = ["testo 890", "testo 885", "testo 883", "testo 872"]

    def __init__(self, model: str = "testo 885"):
        super().__init__(model)
        self._protocol = SimulatedProtocol()

    def get_capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            modalities=["TG"],
            supported_formats=[DataFormat.JPEG, DataFormat.PROPRIETARY],
            max_resolution=(640, 480),
            supports_streaming=True,
            max_frame_rate=33.0,
            features=["SuperResolution", "Thermography App", "Site Recognition"],
        )

    def connect(self, config: ConnectionConfig) -> bool:
        self._connected = self._protocol.connect(config)
        return self._connected

    def disconnect(self) -> bool:
        self._connected = False
        return True

    def acquire_image(self, params: dict = None) -> ImageData:
        return ImageData(
            data=np.random.normal(32, 1.5, (480, 640)).astype(np.float32),
            timestamp=datetime.now(),
            metadata={"vendor": self.VENDOR_NAME},
            modality="thermal",
        )

    def configure_device(self, settings: dict) -> bool:
        return True

    def get_device_status(self) -> dict:
        return {"vendor": self.VENDOR_NAME, "connected": self._connected}

    def export_to_dicom(self, image_data: ImageData) -> bytes:
        return b""

    def import_from_vendor_format(self, data: bytes) -> ImageData:
        return ImageData(data=np.array([]), timestamp=datetime.now(), modality="thermal")
