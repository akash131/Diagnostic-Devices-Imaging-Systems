"""Ultrasound transducer implementation."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import numpy as np

from ..base import ImagingDevice, ImageData, DeviceStatus, DeviceCalibration


class TransducerType(Enum):
    """Types of ultrasound transducers."""

    LINEAR = "linear"
    CONVEX = "convex"
    PHASED_ARRAY = "phased_array"
    ENDOCAVITY = "endocavity"
    MATRIX = "matrix"


@dataclass
class TransducerSpecs:
    """Transducer specifications."""

    frequency_range: tuple[float, float]  # MHz
    element_count: int
    element_pitch: float  # mm
    aperture_size: float  # mm
    focal_depth_range: tuple[float, float]  # mm
    bandwidth: float  # percentage


class TransducerArray:
    """Represents a transducer element array."""

    def __init__(
        self,
        num_elements: int,
        element_pitch: float,
        center_frequency: float,
    ):
        self.num_elements = num_elements
        self.element_pitch = element_pitch
        self.center_frequency = center_frequency
        self._element_positions = self._calculate_positions()
        self._element_delays = np.zeros(num_elements)

    def _calculate_positions(self) -> np.ndarray:
        """Calculate element positions."""
        positions = np.arange(self.num_elements) * self.element_pitch
        return positions - positions.mean()

    def set_focus(self, focal_point: tuple[float, float]) -> np.ndarray:
        """Calculate delays for focusing at a point."""
        fx, fz = focal_point
        c = 1540.0  # Speed of sound in tissue (m/s)

        distances = np.sqrt((self._element_positions - fx) ** 2 + fz**2)
        max_distance = distances.max()
        self._element_delays = (max_distance - distances) / c
        return self._element_delays

    def get_aperture_function(self, apodization: str = "hanning") -> np.ndarray:
        """Get aperture weighting function."""
        n = self.num_elements
        if apodization == "hanning":
            return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))
        elif apodization == "hamming":
            return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
        elif apodization == "rectangular":
            return np.ones(n)
        else:
            raise ValueError(f"Unknown apodization type: {apodization}")


class UltrasoundTransducer(ImagingDevice):
    """Ultrasound transducer device implementation."""

    def __init__(
        self,
        device_id: str,
        device_name: str,
        transducer_type: TransducerType,
        center_frequency: float,
        num_elements: int = 128,
        element_pitch: float = 0.3,
        manufacturer: str = "",
        model: str = "",
    ):
        super().__init__(device_id, device_name, manufacturer, model)
        self.transducer_type = transducer_type
        self.center_frequency = center_frequency
        self.num_elements = num_elements
        self.element_pitch = element_pitch

        self._array = TransducerArray(num_elements, element_pitch, center_frequency)
        self._specs = self._create_default_specs()
        self._gain: float = 1.0
        self._depth: float = 100.0  # mm
        self._focal_zones: list[float] = [50.0]  # mm

    def _create_default_specs(self) -> TransducerSpecs:
        """Create default transducer specifications."""
        freq_min = self.center_frequency * 0.6
        freq_max = self.center_frequency * 1.4
        aperture = self.num_elements * self.element_pitch
        return TransducerSpecs(
            frequency_range=(freq_min, freq_max),
            element_count=self.num_elements,
            element_pitch=self.element_pitch,
            aperture_size=aperture,
            focal_depth_range=(10.0, 200.0),
            bandwidth=80.0,
        )

    def initialize(self) -> bool:
        """Initialize the ultrasound transducer."""
        try:
            self._status = DeviceStatus.INITIALIZING

            # Simulate hardware initialization
            self._verify_transducer_connection()
            self._load_transducer_profile()

            self._status = DeviceStatus.READY
            return True
        except Exception as e:
            self.log_error(f"Initialization failed: {str(e)}")
            return False

    def _verify_transducer_connection(self):
        """Verify transducer hardware connection."""
        # Simulated verification
        pass

    def _load_transducer_profile(self):
        """Load transducer calibration profile."""
        # Simulated profile loading
        pass

    def shutdown(self) -> bool:
        """Shutdown the transducer safely."""
        try:
            self._status = DeviceStatus.OFFLINE
            return True
        except Exception as e:
            self.log_error(f"Shutdown failed: {str(e)}")
            return False

    def acquire(
        self,
        depth: Optional[float] = None,
        gain: Optional[float] = None,
        focal_zones: Optional[list[float]] = None,
        **params,
    ) -> ImageData:
        """Acquire ultrasound image data."""
        if not self.is_ready:
            raise RuntimeError(f"Device not ready. Current status: {self._status}")

        self._status = DeviceStatus.ACQUIRING

        # Update acquisition parameters
        if depth is not None:
            self._depth = depth
        if gain is not None:
            self._gain = gain
        if focal_zones is not None:
            self._focal_zones = focal_zones

        try:
            # Generate simulated RF data
            rf_data = self._generate_rf_data()

            self._last_acquisition = datetime.now()
            self._status = DeviceStatus.READY

            return ImageData(
                data=rf_data,
                timestamp=self._last_acquisition,
                metadata={
                    "modality": "ultrasound",
                    "transducer_type": self.transducer_type.value,
                    "center_frequency_mhz": self.center_frequency,
                    "depth_mm": self._depth,
                    "gain": self._gain,
                    "focal_zones": self._focal_zones,
                    "num_elements": self.num_elements,
                },
                device_id=self.device_id,
                modality="ultrasound",
            )
        except Exception as e:
            self.log_error(f"Acquisition failed: {str(e)}")
            raise

    def _generate_rf_data(self) -> np.ndarray:
        """Generate simulated RF data."""
        # Simulate RF data acquisition
        num_samples = int(self._depth * 13)  # ~13 samples per mm at 40MHz
        num_lines = self.num_elements

        # Generate base RF signal with noise
        rf_data = np.random.randn(num_samples, num_lines) * 0.1

        # Add some simulated tissue echoes
        for _ in range(20):
            depth_idx = np.random.randint(0, num_samples)
            line_idx = np.random.randint(0, num_lines)
            amplitude = np.random.uniform(0.5, 2.0)

            # Create echo with decay
            echo = amplitude * np.exp(-np.abs(np.arange(num_samples) - depth_idx) / 20)
            rf_data[:, line_idx] += echo * np.cos(
                2 * np.pi * self.center_frequency * np.arange(num_samples) / 40
            )

        return rf_data * self._gain

    def calibrate(self, phantom_data: Optional[np.ndarray] = None, **params) -> DeviceCalibration:
        """Calibrate the ultrasound transducer."""
        self._status = DeviceStatus.MAINTENANCE

        try:
            calibration_params = {
                "element_sensitivity": self._calibrate_element_sensitivity(),
                "time_gain_compensation": self._calculate_tgc(),
                "beam_profile": self._measure_beam_profile(),
            }

            self._calibration = DeviceCalibration(
                calibration_date=datetime.now(),
                calibration_parameters=calibration_params,
                is_valid=True,
                expiry_date=datetime.now() + timedelta(days=30),
            )

            self._status = DeviceStatus.READY
            return self._calibration
        except Exception as e:
            self.log_error(f"Calibration failed: {str(e)}")
            raise

    def _calibrate_element_sensitivity(self) -> np.ndarray:
        """Calibrate individual element sensitivity."""
        # Simulated sensitivity calibration
        return np.ones(self.num_elements) + np.random.randn(self.num_elements) * 0.05

    def _calculate_tgc(self) -> np.ndarray:
        """Calculate time-gain compensation curve."""
        depths = np.linspace(0, self._depth, 100)
        attenuation = 0.5  # dB/cm/MHz
        tgc = np.exp(attenuation * self.center_frequency * depths / 10)
        return tgc / tgc.max()

    def _measure_beam_profile(self) -> dict:
        """Measure transducer beam profile."""
        return {
            "lateral_resolution": 0.5,  # mm
            "axial_resolution": 0.3,  # mm
            "beam_width_6db": 2.0,  # mm at focal depth
        }

    def set_imaging_mode(self, mode: str):
        """Set imaging mode (B-mode, M-mode, Doppler, etc.)."""
        valid_modes = ["b_mode", "m_mode", "doppler", "color_doppler", "power_doppler"]
        if mode.lower() not in valid_modes:
            raise ValueError(f"Invalid imaging mode. Must be one of: {valid_modes}")
        # Mode setting would be implemented here

    @property
    def specs(self) -> TransducerSpecs:
        """Get transducer specifications."""
        return self._specs
