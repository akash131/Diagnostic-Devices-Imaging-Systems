"""Vendor adapters for X-ray equipment manufacturers."""

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


class GEHealthcareXRay(VendorAdapter):
    """Adapter for GE Healthcare X-ray systems."""

    VENDOR_NAME = "GE Healthcare"
    VENDOR_ID = "ge_healthcare_xray"
    SUPPORTED_MODELS = [
        "Optima XR240amx",
        "Optima XR220amx",
        "Discovery XR656 Plus",
        "Definium 656",
        "AMX 240",
    ]

    def __init__(self, model: str = "Optima XR240amx"):
        super().__init__(model)
        self._protocol = SimulatedProtocol()
        self._aec_enabled = True
        self._grid_in = True

    def get_capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            modalities=["DX", "CR"],
            supported_formats=[DataFormat.DICOM, DataFormat.RAW],
            max_resolution=(3072, 3072),
            supports_streaming=False,
            supports_remote_control=True,
            supports_calibration=True,
            supports_dicom_export=True,
            max_frame_rate=2.0,
            color_depth=14,
            features=[
                "Auto Exposure Control",
                "Dual Energy Subtraction",
                "Image Stitching",
                "Virtual Grid",
                "Dose Tracking",
                "Portable/Mobile",
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

        kvp = params.get("kvp", 80)
        ma = params.get("ma", 200)
        exposure_time = params.get("exposure_time", 0.05)
        body_part = params.get("body_part", "CHEST")

        # Generate simulated X-ray image
        resolution = (2048, 2048)
        image = self._generate_xray_image(resolution, kvp, ma * exposure_time)

        return ImageData(
            data=image,
            timestamp=datetime.now(),
            metadata={
                "vendor": self.VENDOR_NAME,
                "model": self.model,
                "modality": "xray",
                "kvp": kvp,
                "ma": ma,
                "exposure_time_s": exposure_time,
                "mas": ma * exposure_time,
                "body_part": body_part,
                "aec_enabled": self._aec_enabled,
                "grid": "IN" if self._grid_in else "OUT",
            },
            device_id=f"{self.VENDOR_ID}_{self.model}",
            modality="xray",
        )

    def _generate_xray_image(
        self,
        resolution: tuple[int, int],
        kvp: float,
        mas: float,
    ) -> np.ndarray:
        """Generate simulated X-ray image."""
        height, width = resolution

        # Base exposure level
        exposure = kvp * mas / 8000
        base_intensity = exposure * 10000

        # Generate with Poisson noise
        image = np.random.poisson(base_intensity, (height, width)).astype(np.float64)

        # Add anatomical structures
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2

        # Lung fields (darker areas)
        left_lung = ((x - center_x + 200) ** 2 / 150 ** 2 + (y - center_y) ** 2 / 300 ** 2) < 1
        right_lung = ((x - center_x - 200) ** 2 / 150 ** 2 + (y - center_y) ** 2 / 300 ** 2) < 1
        image[left_lung] *= 1.3
        image[right_lung] *= 1.3

        # Heart shadow
        heart = ((x - center_x + 50) ** 2 / 100 ** 2 + (y - center_y + 100) ** 2 / 120 ** 2) < 1
        image[heart] *= 0.6

        # Spine
        spine = np.abs(x - center_x) < 30
        image[spine] *= 0.5

        # Log transform and normalize to 14-bit
        image = np.log1p(image)
        image = (image - image.min()) / (image.max() - image.min()) * 16383

        return image.astype(np.uint16)

    def configure_device(self, settings: dict) -> bool:
        if "aec" in settings:
            self._aec_enabled = settings["aec"]
        if "grid" in settings:
            self._grid_in = settings["grid"]
        return True

    def get_device_status(self) -> dict:
        return {
            "vendor": self.VENDOR_NAME,
            "model": self.model,
            "connected": self._connected,
            "aec_enabled": self._aec_enabled,
            "grid_in": self._grid_in,
            "ready": self._connected,
        }

    def export_to_dicom(self, image_data: ImageData) -> bytes:
        dataset = ModalityBuilder.create_xray_dataset(
            image_data.data,
            "Anonymous",
            "000000",
            kvp=image_data.metadata.get("kvp", 80),
            exposure_time_ms=int(image_data.metadata.get("exposure_time_s", 0.05) * 1000),
            tube_current_ma=image_data.metadata.get("ma", 200),
            body_part=image_data.metadata.get("body_part", "CHEST"),
            manufacturer=self.VENDOR_NAME,
            model=self.model,
        )
        return str(dataset.to_dict()).encode()

    def import_from_vendor_format(self, data: bytes) -> ImageData:
        return ImageData(
            data=np.frombuffer(data, dtype=np.uint16),
            timestamp=datetime.now(),
            metadata={"vendor": self.VENDOR_NAME},
            modality="xray",
        )

    def get_supported_settings(self) -> dict:
        return {
            "kvp": {"min": 40, "max": 150, "unit": "kV"},
            "ma": {"min": 10, "max": 500, "unit": "mA"},
            "exposure_time": {"min": 0.001, "max": 5.0, "unit": "s"},
            "sfd": {"min": 100, "max": 200, "unit": "cm"},
            "body_parts": ["CHEST", "ABDOMEN", "PELVIS", "SPINE", "EXTREMITY", "SKULL"],
        }


class PhilipsXRay(VendorAdapter):
    """Adapter for Philips X-ray systems."""

    VENDOR_NAME = "Philips Healthcare"
    VENDOR_ID = "philips_xray"
    SUPPORTED_MODELS = [
        "DigitalDiagnost C90",
        "MobileDiagnost wDR",
        "ProxiDiagnost N90",
    ]

    def __init__(self, model: str = "DigitalDiagnost C90"):
        super().__init__(model)
        self._protocol = SimulatedProtocol()

    def get_capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            modalities=["DX"],
            supported_formats=[DataFormat.DICOM],
            max_resolution=(3000, 3000),
            supports_remote_control=True,
            color_depth=14,
            features=["SkyPlate", "Eleva", "UNIQUE imaging"],
        )

    def connect(self, config: ConnectionConfig) -> bool:
        self._connected = self._protocol.connect(config)
        return self._connected

    def disconnect(self) -> bool:
        self._connected = False
        return True

    def acquire_image(self, params: dict = None) -> ImageData:
        return ImageData(
            data=np.random.poisson(8000, (2048, 2048)).astype(np.uint16),
            timestamp=datetime.now(),
            metadata={"vendor": self.VENDOR_NAME},
            modality="xray",
        )

    def configure_device(self, settings: dict) -> bool:
        return True

    def get_device_status(self) -> dict:
        return {"vendor": self.VENDOR_NAME, "connected": self._connected}

    def export_to_dicom(self, image_data: ImageData) -> bytes:
        return b""

    def import_from_vendor_format(self, data: bytes) -> ImageData:
        return ImageData(data=np.array([]), timestamp=datetime.now(), modality="xray")


class SiemensXRay(VendorAdapter):
    """Adapter for Siemens X-ray systems."""

    VENDOR_NAME = "Siemens Healthineers"
    VENDOR_ID = "siemens_xray"
    SUPPORTED_MODELS = [
        "MOBILETT Mira Max",
        "MOBILETT Elara Max",
        "Ysio Max",
        "Multix Fusion Max",
    ]

    def __init__(self, model: str = "MOBILETT Mira Max"):
        super().__init__(model)
        self._protocol = SimulatedProtocol()

    def get_capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            modalities=["DX"],
            supported_formats=[DataFormat.DICOM],
            max_resolution=(3072, 3072),
            color_depth=14,
            features=["MAX Technology", "Retina Display", "syngo imaging"],
        )

    def connect(self, config: ConnectionConfig) -> bool:
        self._connected = self._protocol.connect(config)
        return self._connected

    def disconnect(self) -> bool:
        self._connected = False
        return True

    def acquire_image(self, params: dict = None) -> ImageData:
        return ImageData(
            data=np.random.poisson(9000, (2048, 2048)).astype(np.uint16),
            timestamp=datetime.now(),
            metadata={"vendor": self.VENDOR_NAME},
            modality="xray",
        )

    def configure_device(self, settings: dict) -> bool:
        return True

    def get_device_status(self) -> dict:
        return {"vendor": self.VENDOR_NAME, "connected": self._connected}

    def export_to_dicom(self, image_data: ImageData) -> bytes:
        return b""

    def import_from_vendor_format(self, data: bytes) -> ImageData:
        return ImageData(data=np.array([]), timestamp=datetime.now(), modality="xray")


class CareStreamXRay(VendorAdapter):
    """Adapter for Carestream X-ray systems."""

    VENDOR_NAME = "Carestream Health"
    VENDOR_ID = "carestream_xray"
    SUPPORTED_MODELS = ["DRX-Revolution", "DRX-Evolution Plus", "DRX-Compass"]

    def __init__(self, model: str = "DRX-Revolution"):
        super().__init__(model)
        self._protocol = SimulatedProtocol()

    def get_capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            modalities=["DX", "CR"],
            supported_formats=[DataFormat.DICOM],
            max_resolution=(3520, 4280),
            color_depth=14,
            features=["DRX Detectors", "ImageView Software", "Tube Navigation"],
        )

    def connect(self, config: ConnectionConfig) -> bool:
        self._connected = self._protocol.connect(config)
        return self._connected

    def disconnect(self) -> bool:
        self._connected = False
        return True

    def acquire_image(self, params: dict = None) -> ImageData:
        return ImageData(
            data=np.random.poisson(8500, (2048, 2048)).astype(np.uint16),
            timestamp=datetime.now(),
            metadata={"vendor": self.VENDOR_NAME},
            modality="xray",
        )

    def configure_device(self, settings: dict) -> bool:
        return True

    def get_device_status(self) -> dict:
        return {"vendor": self.VENDOR_NAME, "connected": self._connected}

    def export_to_dicom(self, image_data: ImageData) -> bytes:
        return b""

    def import_from_vendor_format(self, data: bytes) -> ImageData:
        return ImageData(data=np.array([]), timestamp=datetime.now(), modality="xray")


class FujifilmXRay(VendorAdapter):
    """Adapter for Fujifilm X-ray systems."""

    VENDOR_NAME = "Fujifilm"
    VENDOR_ID = "fujifilm_xray"
    SUPPORTED_MODELS = ["FDR D-EVO II", "FDR Go PLUS", "FDR Visionary Suite"]

    def __init__(self, model: str = "FDR D-EVO II"):
        super().__init__(model)
        self._protocol = SimulatedProtocol()

    def get_capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            modalities=["DX", "CR"],
            supported_formats=[DataFormat.DICOM],
            max_resolution=(3072, 3072),
            color_depth=14,
            features=["ISS Technology", "Virtual Grid", "Dynamic Visualization"],
        )

    def connect(self, config: ConnectionConfig) -> bool:
        self._connected = self._protocol.connect(config)
        return self._connected

    def disconnect(self) -> bool:
        self._connected = False
        return True

    def acquire_image(self, params: dict = None) -> ImageData:
        return ImageData(
            data=np.random.poisson(8200, (2048, 2048)).astype(np.uint16),
            timestamp=datetime.now(),
            metadata={"vendor": self.VENDOR_NAME},
            modality="xray",
        )

    def configure_device(self, settings: dict) -> bool:
        return True

    def get_device_status(self) -> dict:
        return {"vendor": self.VENDOR_NAME, "connected": self._connected}

    def export_to_dicom(self, image_data: ImageData) -> bytes:
        return b""

    def import_from_vendor_format(self, data: bytes) -> ImageData:
        return ImageData(data=np.array([]), timestamp=datetime.now(), modality="xray")
