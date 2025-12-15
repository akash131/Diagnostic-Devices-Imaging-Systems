"""Vendor adapters for OCT equipment manufacturers."""

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


class ZeissOCT(VendorAdapter):
    """Adapter for Zeiss OCT systems."""

    VENDOR_NAME = "Carl Zeiss Meditec"
    VENDOR_ID = "zeiss_oct"
    SUPPORTED_MODELS = [
        "CIRRUS HD-OCT 6000",
        "CIRRUS HD-OCT 5000",
        "PLEX Elite 9000",
        "PRIMUS 200",
    ]

    def __init__(self, model: str = "CIRRUS HD-OCT 6000"):
        super().__init__(model)
        self._protocol = SimulatedProtocol()
        self._scan_pattern = "macular_cube"
        self._eye = "OD"

    def get_capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            modalities=["OPT"],
            supported_formats=[DataFormat.DICOM, DataFormat.PROPRIETARY],
            max_resolution=(1024, 512),
            supports_streaming=False,
            supports_remote_control=True,
            supports_calibration=True,
            supports_dicom_export=True,
            max_frame_rate=100000,  # A-scans per second
            color_depth=8,
            features=[
                "HD Imaging",
                "FastTrac",
                "AutoCenter",
                "RNFL Analysis",
                "Ganglion Cell Analysis",
                "Macular Thickness",
                "OCTA",
                "Advanced RPE Analysis",
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

        scan_pattern = params.get("scan_pattern", self._scan_pattern)
        eye = params.get("eye", self._eye)

        # Generate simulated OCT image based on scan pattern
        if scan_pattern == "macular_cube":
            image = self._generate_macular_cube()
        elif scan_pattern == "optic_disc_cube":
            image = self._generate_optic_disc_cube()
        elif scan_pattern == "hd_5_line":
            image = self._generate_hd_line_scan()
        else:
            image = self._generate_macular_cube()

        return ImageData(
            data=image,
            timestamp=datetime.now(),
            metadata={
                "vendor": self.VENDOR_NAME,
                "model": self.model,
                "modality": "oct",
                "scan_pattern": scan_pattern,
                "eye": eye,
                "wavelength_nm": 840,
                "axial_resolution_um": 5,
                "transverse_resolution_um": 15,
            },
            device_id=f"{self.VENDOR_ID}_{self.model}",
            modality="oct",
        )

    def _generate_macular_cube(self) -> np.ndarray:
        """Generate simulated macular cube scan."""
        # 512 x 128 x 1024 (A-scans x B-scans x depth)
        depth, width, num_bscans = 1024, 512, 128
        volume = np.zeros((depth, width, num_bscans), dtype=np.uint8)

        for b in range(num_bscans):
            # Create layered retinal structure
            x = np.arange(width)

            # ILM surface with foveal depression
            ilm = 100 + 20 * np.exp(-((x - width // 2) ** 2) / 5000)

            # RPE surface
            rpe = 300 + 15 * np.exp(-((x - width // 2) ** 2) / 8000)

            for i in range(width):
                # Layers
                volume[int(ilm[i]):int(ilm[i]) + 30, i, b] = 180  # RNFL
                volume[int(ilm[i]) + 30:int(ilm[i]) + 80, i, b] = 120  # Inner layers
                volume[int(ilm[i]) + 80:int(rpe[i]) - 50, i, b] = 80  # ONL
                volume[int(rpe[i]) - 50:int(rpe[i]), i, b] = 200  # Photoreceptors
                volume[int(rpe[i]):int(rpe[i]) + 20, i, b] = 255  # RPE

        # Add speckle noise
        noise = np.random.rayleigh(scale=10, size=volume.shape)
        volume = np.clip(volume + noise, 0, 255).astype(np.uint8)

        return volume

    def _generate_optic_disc_cube(self) -> np.ndarray:
        """Generate simulated optic disc cube scan."""
        depth, width, num_bscans = 1024, 200, 200
        volume = np.zeros((depth, width, num_bscans), dtype=np.uint8)

        # Simplified optic disc structure
        for b in range(num_bscans):
            for i in range(width):
                dist_from_center = np.sqrt((i - width // 2) ** 2 + (b - num_bscans // 2) ** 2)

                # Cup depression in center
                if dist_from_center < 30:
                    surface = 150 + int(dist_from_center * 2)
                else:
                    surface = 150

                volume[surface:surface + 100, i, b] = np.linspace(200, 50, 100)

        return volume.astype(np.uint8)

    def _generate_hd_line_scan(self) -> np.ndarray:
        """Generate high-definition line scan."""
        depth, width = 1024, 4096

        image = np.zeros((depth, width), dtype=np.uint8)
        x = np.arange(width)

        # High-resolution single B-scan
        ilm = 100 + 30 * np.sin(x / 200) + 20 * np.exp(-((x - width // 2) ** 2) / 100000)

        for i in range(width):
            image[int(ilm[i]):int(ilm[i]) + 200, i] = np.linspace(200, 50, 200)

        return image

    def configure_device(self, settings: dict) -> bool:
        if "scan_pattern" in settings:
            self._scan_pattern = settings["scan_pattern"]
        if "eye" in settings:
            self._eye = settings["eye"]
        return True

    def get_device_status(self) -> dict:
        return {
            "vendor": self.VENDOR_NAME,
            "model": self.model,
            "connected": self._connected,
            "scan_pattern": self._scan_pattern,
            "eye": self._eye,
        }

    def export_to_dicom(self, image_data: ImageData) -> bytes:
        # For 3D data, export middle slice
        if image_data.data.ndim == 3:
            slice_data = image_data.data[:, :, image_data.data.shape[2] // 2]
        else:
            slice_data = image_data.data

        dataset = ModalityBuilder.create_oct_dataset(
            slice_data,
            "Anonymous",
            "000000",
            wavelength_nm=image_data.metadata.get("wavelength_nm", 840),
            manufacturer=self.VENDOR_NAME,
            model=self.model,
        )
        return str(dataset.to_dict()).encode()

    def import_from_vendor_format(self, data: bytes) -> ImageData:
        return ImageData(
            data=np.frombuffer(data, dtype=np.uint8),
            timestamp=datetime.now(),
            metadata={"vendor": self.VENDOR_NAME, "format": "zeiss_raw"},
            modality="oct",
        )

    def get_supported_settings(self) -> dict:
        return {
            "scan_patterns": [
                "macular_cube",
                "optic_disc_cube",
                "hd_5_line",
                "hd_21_line",
                "anterior_segment",
            ],
            "eye": ["OD", "OS"],
            "signal_strength_threshold": {"min": 1, "max": 10},
        }


class HeidelbergOCT(VendorAdapter):
    """Adapter for Heidelberg Engineering OCT systems."""

    VENDOR_NAME = "Heidelberg Engineering"
    VENDOR_ID = "heidelberg_oct"
    SUPPORTED_MODELS = ["SPECTRALIS", "SPECTRALIS HRA+OCT", "SPECTRALIS OCT2"]

    def __init__(self, model: str = "SPECTRALIS"):
        super().__init__(model)
        self._protocol = SimulatedProtocol()

    def get_capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            modalities=["OPT"],
            supported_formats=[DataFormat.DICOM, DataFormat.PROPRIETARY],
            max_resolution=(1536, 496),
            supports_streaming=False,
            max_frame_rate=85000,
            features=[
                "TruTrack",
                "AutoRescan",
                "Multicolor Imaging",
                "OCTA",
                "Glaucoma Module",
                "OCT-A Module",
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
            data=np.random.rayleigh(scale=50, size=(496, 1536)).astype(np.uint8),
            timestamp=datetime.now(),
            metadata={"vendor": self.VENDOR_NAME, "model": self.model},
            modality="oct",
        )

    def configure_device(self, settings: dict) -> bool:
        return True

    def get_device_status(self) -> dict:
        return {"vendor": self.VENDOR_NAME, "connected": self._connected}

    def export_to_dicom(self, image_data: ImageData) -> bytes:
        return b""

    def import_from_vendor_format(self, data: bytes) -> ImageData:
        return ImageData(data=np.array([]), timestamp=datetime.now(), modality="oct")


class TopconOCT(VendorAdapter):
    """Adapter for Topcon OCT systems."""

    VENDOR_NAME = "Topcon"
    VENDOR_ID = "topcon_oct"
    SUPPORTED_MODELS = ["DRI OCT Triton", "Maestro2", "3D OCT-1 Maestro"]

    def __init__(self, model: str = "DRI OCT Triton"):
        super().__init__(model)
        self._protocol = SimulatedProtocol()

    def get_capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            modalities=["OPT"],
            supported_formats=[DataFormat.DICOM],
            max_resolution=(1024, 512),
            features=["Swept Source", "Wide Field", "OCTA", "Hood Report"],
        )

    def connect(self, config: ConnectionConfig) -> bool:
        self._connected = self._protocol.connect(config)
        return self._connected

    def disconnect(self) -> bool:
        self._connected = False
        return True

    def acquire_image(self, params: dict = None) -> ImageData:
        return ImageData(
            data=np.random.rayleigh(scale=45, size=(512, 1024)).astype(np.uint8),
            timestamp=datetime.now(),
            metadata={"vendor": self.VENDOR_NAME},
            modality="oct",
        )

    def configure_device(self, settings: dict) -> bool:
        return True

    def get_device_status(self) -> dict:
        return {"vendor": self.VENDOR_NAME, "connected": self._connected}

    def export_to_dicom(self, image_data: ImageData) -> bytes:
        return b""

    def import_from_vendor_format(self, data: bytes) -> ImageData:
        return ImageData(data=np.array([]), timestamp=datetime.now(), modality="oct")


class OptovueOCT(VendorAdapter):
    """Adapter for Optovue OCT systems."""

    VENDOR_NAME = "Optovue"
    VENDOR_ID = "optovue_oct"
    SUPPORTED_MODELS = ["Avanti RTVue XR", "AngioVue", "iVue"]

    def __init__(self, model: str = "Avanti RTVue XR"):
        super().__init__(model)
        self._protocol = SimulatedProtocol()

    def get_capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            modalities=["OPT"],
            supported_formats=[DataFormat.DICOM],
            max_resolution=(1024, 640),
            features=["AngioVue OCTA", "En Face", "Cross Line"],
        )

    def connect(self, config: ConnectionConfig) -> bool:
        self._connected = self._protocol.connect(config)
        return self._connected

    def disconnect(self) -> bool:
        self._connected = False
        return True

    def acquire_image(self, params: dict = None) -> ImageData:
        return ImageData(
            data=np.random.rayleigh(scale=40, size=(640, 1024)).astype(np.uint8),
            timestamp=datetime.now(),
            metadata={"vendor": self.VENDOR_NAME},
            modality="oct",
        )

    def configure_device(self, settings: dict) -> bool:
        return True

    def get_device_status(self) -> dict:
        return {"vendor": self.VENDOR_NAME, "connected": self._connected}

    def export_to_dicom(self, image_data: ImageData) -> bytes:
        return b""

    def import_from_vendor_format(self, data: bytes) -> ImageData:
        return ImageData(data=np.array([]), timestamp=datetime.now(), modality="oct")
