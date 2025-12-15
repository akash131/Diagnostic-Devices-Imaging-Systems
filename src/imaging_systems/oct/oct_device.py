"""Optical Coherence Tomography device implementation."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import numpy as np

from ..base import ImagingDevice, ImageData, DeviceStatus, DeviceCalibration


class OCTType(Enum):
    """Types of OCT systems."""

    TIME_DOMAIN = "time_domain"
    SPECTRAL_DOMAIN = "spectral_domain"
    SWEPT_SOURCE = "swept_source"


class ScanPattern(Enum):
    """OCT scan patterns."""

    B_SCAN = "b_scan"  # Single cross-section
    VOLUME = "volume"  # 3D volume
    RADIAL = "radial"  # Star pattern
    RASTER = "raster"  # Rectangular grid
    CIRCULAR = "circular"  # Circle scan (for optic nerve)


@dataclass
class OCTSpecs:
    """OCT system specifications."""

    oct_type: OCTType
    center_wavelength: float  # nm
    bandwidth: float  # nm
    axial_resolution: float  # μm
    lateral_resolution: float  # μm
    scan_depth: float  # mm
    scan_rate: float  # A-scans per second


@dataclass
class LightSource:
    """OCT light source parameters."""

    source_type: str  # SLD, swept laser, etc.
    center_wavelength: float  # nm
    bandwidth: float  # nm
    power: float  # mW
    coherence_length: float  # μm


class OCTDevice(ImagingDevice):
    """Optical Coherence Tomography device implementation."""

    def __init__(
        self,
        device_id: str,
        device_name: str,
        oct_type: OCTType = OCTType.SPECTRAL_DOMAIN,
        center_wavelength: float = 840.0,
        bandwidth: float = 50.0,
        manufacturer: str = "",
        model: str = "",
    ):
        super().__init__(device_id, device_name, manufacturer, model)
        self.oct_type = oct_type
        self.center_wavelength = center_wavelength
        self.bandwidth = bandwidth

        self._light_source = self._create_light_source()
        self._specs = self._calculate_specs()
        self._reference_arm_position: float = 0.0
        self._scan_pattern = ScanPattern.B_SCAN
        self._averaging: int = 1

    def _create_light_source(self) -> LightSource:
        """Create light source based on OCT type."""
        if self.oct_type == OCTType.SPECTRAL_DOMAIN:
            return LightSource(
                source_type="SLD",
                center_wavelength=self.center_wavelength,
                bandwidth=self.bandwidth,
                power=5.0,
                coherence_length=self._calculate_coherence_length(),
            )
        elif self.oct_type == OCTType.SWEPT_SOURCE:
            return LightSource(
                source_type="Swept Laser",
                center_wavelength=1050.0,
                bandwidth=100.0,
                power=20.0,
                coherence_length=self._calculate_coherence_length(),
            )
        else:
            return LightSource(
                source_type="SLD",
                center_wavelength=self.center_wavelength,
                bandwidth=self.bandwidth,
                power=2.0,
                coherence_length=self._calculate_coherence_length(),
            )

    def _calculate_coherence_length(self) -> float:
        """Calculate coherence length in μm."""
        # Coherence length = (2 * ln(2) / π) * (λ² / Δλ)
        return (2 * np.log(2) / np.pi) * (self.center_wavelength**2 / self.bandwidth)

    def _calculate_specs(self) -> OCTSpecs:
        """Calculate OCT system specifications."""
        coherence_length = self._calculate_coherence_length()

        # Axial resolution ≈ coherence length
        axial_res = coherence_length / 1000  # Convert to μm

        # Lateral resolution depends on optics (typically 15-20 μm)
        lateral_res = 15.0

        # Scan depth depends on spectrometer for SD-OCT
        if self.oct_type == OCTType.SPECTRAL_DOMAIN:
            scan_depth = 2.5  # mm
            scan_rate = 70000  # A-scans/s
        elif self.oct_type == OCTType.SWEPT_SOURCE:
            scan_depth = 6.0  # mm
            scan_rate = 200000  # A-scans/s
        else:
            scan_depth = 2.0  # mm
            scan_rate = 400  # A-scans/s

        return OCTSpecs(
            oct_type=self.oct_type,
            center_wavelength=self.center_wavelength,
            bandwidth=self.bandwidth,
            axial_resolution=axial_res,
            lateral_resolution=lateral_res,
            scan_depth=scan_depth,
            scan_rate=scan_rate,
        )

    def initialize(self) -> bool:
        """Initialize the OCT device."""
        try:
            self._status = DeviceStatus.INITIALIZING

            # Initialize light source
            self._initialize_light_source()

            # Initialize spectrometer/detector
            self._initialize_detector()

            # Set reference arm position
            self._optimize_reference_arm()

            self._status = DeviceStatus.READY
            return True
        except Exception as e:
            self.log_error(f"Initialization failed: {str(e)}")
            return False

    def _initialize_light_source(self):
        """Initialize the light source."""
        # Simulated light source initialization
        pass

    def _initialize_detector(self):
        """Initialize spectrometer or detector."""
        # Simulated detector initialization
        pass

    def _optimize_reference_arm(self):
        """Optimize reference arm position."""
        # Auto-adjustment of reference arm for optimal signal
        self._reference_arm_position = 0.5

    def shutdown(self) -> bool:
        """Shutdown the OCT device."""
        try:
            # Turn off light source
            self._status = DeviceStatus.OFFLINE
            return True
        except Exception as e:
            self.log_error(f"Shutdown failed: {str(e)}")
            return False

    def acquire(
        self,
        scan_pattern: Optional[ScanPattern] = None,
        scan_width: float = 6.0,
        scan_depth: float = 2.0,
        num_a_scans: int = 512,
        num_b_scans: int = 1,
        averaging: int = 1,
        **params,
    ) -> ImageData:
        """Acquire OCT scan data."""
        if not self.is_ready:
            raise RuntimeError(f"Device not ready. Current status: {self._status}")

        if scan_pattern is not None:
            self._scan_pattern = scan_pattern

        self._status = DeviceStatus.ACQUIRING

        try:
            # Generate OCT data based on scan pattern
            if self._scan_pattern == ScanPattern.B_SCAN:
                data = self._acquire_b_scan(num_a_scans, scan_width, scan_depth, averaging)
            elif self._scan_pattern == ScanPattern.VOLUME:
                data = self._acquire_volume(num_a_scans, num_b_scans, scan_width, scan_depth)
            elif self._scan_pattern == ScanPattern.CIRCULAR:
                data = self._acquire_circular(num_a_scans, scan_width, averaging)
            else:
                data = self._acquire_b_scan(num_a_scans, scan_width, scan_depth, averaging)

            self._last_acquisition = datetime.now()
            self._status = DeviceStatus.READY

            return ImageData(
                data=data,
                timestamp=self._last_acquisition,
                metadata={
                    "modality": "oct",
                    "oct_type": self.oct_type.value,
                    "scan_pattern": self._scan_pattern.value,
                    "scan_width_mm": scan_width,
                    "scan_depth_mm": scan_depth,
                    "num_a_scans": num_a_scans,
                    "num_b_scans": num_b_scans,
                    "averaging": averaging,
                    "center_wavelength_nm": self.center_wavelength,
                    "axial_resolution_um": self._specs.axial_resolution,
                    "lateral_resolution_um": self._specs.lateral_resolution,
                },
                device_id=self.device_id,
                modality="oct",
            )
        except Exception as e:
            self.log_error(f"Acquisition failed: {str(e)}")
            raise

    def _acquire_b_scan(
        self,
        num_a_scans: int,
        scan_width: float,
        scan_depth: float,
        averaging: int,
    ) -> np.ndarray:
        """Acquire a single B-scan (cross-section)."""
        depth_samples = int(scan_depth * 500)  # ~500 samples per mm

        # Generate simulated OCT data
        b_scan = np.zeros((depth_samples, num_a_scans))

        for _ in range(averaging):
            # Add tissue layers
            layer_data = self._generate_tissue_layers(depth_samples, num_a_scans)
            b_scan += layer_data

        b_scan /= averaging

        # Add speckle noise
        speckle = np.random.rayleigh(scale=0.3, size=b_scan.shape)
        b_scan *= speckle

        return b_scan

    def _acquire_volume(
        self,
        num_a_scans: int,
        num_b_scans: int,
        scan_width: float,
        scan_depth: float,
    ) -> np.ndarray:
        """Acquire 3D volume scan."""
        depth_samples = int(scan_depth * 500)
        volume = np.zeros((depth_samples, num_a_scans, num_b_scans))

        for i in range(num_b_scans):
            volume[:, :, i] = self._acquire_b_scan(num_a_scans, scan_width, scan_depth, 1)

        return volume

    def _acquire_circular(
        self,
        num_a_scans: int,
        diameter: float,
        averaging: int,
    ) -> np.ndarray:
        """Acquire circular scan (e.g., around optic nerve)."""
        depth_samples = int(2.0 * 500)  # 2mm depth

        circular_scan = np.zeros((depth_samples, num_a_scans))

        for _ in range(averaging):
            # Simulate RNFL thickness variation around optic nerve
            angles = np.linspace(0, 2 * np.pi, num_a_scans, endpoint=False)
            rnfl_thickness = 100 + 30 * np.sin(2 * angles)  # μm

            for i, thickness in enumerate(rnfl_thickness):
                # Create A-scan with RNFL layer
                a_scan = np.zeros(depth_samples)
                surface_idx = 50
                rnfl_end = surface_idx + int(thickness / 2)

                # Surface reflection
                a_scan[surface_idx : surface_idx + 5] = 1.0

                # RNFL layer
                a_scan[surface_idx:rnfl_end] = 0.7 * np.exp(
                    -np.arange(rnfl_end - surface_idx) / 50
                )

                # Deeper layers
                a_scan[rnfl_end:] = 0.3 * np.exp(-np.arange(depth_samples - rnfl_end) / 100)

                circular_scan[:, i] = a_scan

        circular_scan /= averaging

        # Add speckle
        speckle = np.random.rayleigh(scale=0.3, size=circular_scan.shape)
        circular_scan *= speckle

        return circular_scan

    def _generate_tissue_layers(
        self,
        depth_samples: int,
        num_a_scans: int,
    ) -> np.ndarray:
        """Generate simulated tissue layer structure."""
        image = np.zeros((depth_samples, num_a_scans))

        # Define layer positions with some curvature
        x = np.arange(num_a_scans)
        curvature = 20 * np.sin(np.pi * x / num_a_scans)

        # Layer 1: Surface (e.g., epithelium)
        layer1_depth = 30 + curvature
        layer1_thickness = 20

        # Layer 2: Middle layer
        layer2_depth = 80 + curvature
        layer2_thickness = 40

        # Layer 3: Deep layer
        layer3_depth = 200 + curvature
        layer3_thickness = 100

        for i in range(num_a_scans):
            # Layer 1
            start = int(max(0, layer1_depth[i]))
            end = int(min(depth_samples, start + layer1_thickness))
            image[start:end, i] = 0.9

            # Layer 2
            start = int(max(0, layer2_depth[i]))
            end = int(min(depth_samples, start + layer2_thickness))
            image[start:end, i] = 0.6

            # Layer 3
            start = int(max(0, layer3_depth[i]))
            end = int(min(depth_samples, start + layer3_thickness))
            image[start:end, i] = 0.4

        return image

    def calibrate(self, **params) -> DeviceCalibration:
        """Calibrate the OCT device."""
        self._status = DeviceStatus.MAINTENANCE

        try:
            calibration_params = {
                "dispersion_compensation": self._calibrate_dispersion(),
                "spectral_calibration": self._calibrate_spectrometer(),
                "reference_arm_position": self._reference_arm_position,
                "sensitivity": self._measure_sensitivity(),
            }

            self._calibration = DeviceCalibration(
                calibration_date=datetime.now(),
                calibration_parameters=calibration_params,
                is_valid=True,
                expiry_date=datetime.now() + timedelta(days=90),
            )

            self._status = DeviceStatus.READY
            return self._calibration
        except Exception as e:
            self.log_error(f"Calibration failed: {str(e)}")
            raise

    def _calibrate_dispersion(self) -> np.ndarray:
        """Calibrate dispersion compensation."""
        # Generate dispersion coefficients
        return np.array([0.0, 0.0, 1e-6, 0.0])

    def _calibrate_spectrometer(self) -> dict:
        """Calibrate spectrometer wavelength mapping."""
        return {
            "wavelength_start": self.center_wavelength - self.bandwidth / 2,
            "wavelength_end": self.center_wavelength + self.bandwidth / 2,
            "num_pixels": 2048,
            "linearity": 0.999,
        }

    def _measure_sensitivity(self) -> float:
        """Measure system sensitivity in dB."""
        # Typical SD-OCT sensitivity
        return 95.0  # dB

    def set_reference_arm_position(self, position: float):
        """Set reference arm position manually."""
        if not (0 <= position <= 1):
            raise ValueError("Position must be between 0 and 1")
        self._reference_arm_position = position

    @property
    def specs(self) -> OCTSpecs:
        """Get OCT specifications."""
        return self._specs

    @property
    def light_source(self) -> LightSource:
        """Get light source parameters."""
        return self._light_source
