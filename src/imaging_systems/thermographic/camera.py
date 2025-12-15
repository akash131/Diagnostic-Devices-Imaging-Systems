"""Thermographic camera implementation for medical imaging."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import numpy as np

from ..base import ImagingDevice, ImageData, DeviceStatus, DeviceCalibration


class SensorType(Enum):
    """Thermal sensor types."""

    MICROBOLOMETER = "microbolometer"
    PHOTON_DETECTOR = "photon_detector"
    QWIP = "qwip"  # Quantum Well Infrared Photodetector
    MCT = "mct"  # Mercury Cadmium Telluride


@dataclass
class ThermalSensor:
    """Thermal sensor specifications."""

    sensor_type: SensorType
    resolution: tuple[int, int]
    pixel_pitch: float  # μm
    netd: float  # Noise Equivalent Temperature Difference (mK)
    spectral_range: tuple[float, float]  # μm
    frame_rate: float  # Hz
    fpa_temperature: Optional[float] = None  # K (for cooled sensors)


@dataclass
class ThermalImageData:
    """Extended thermal image data with temperature information."""

    raw_data: np.ndarray
    temperature_map: np.ndarray  # Celsius
    min_temp: float
    max_temp: float
    ambient_temp: float
    emissivity: float


class ThermographicCamera(ImagingDevice):
    """Thermographic camera for medical inflammation detection."""

    def __init__(
        self,
        device_id: str,
        device_name: str,
        sensor_type: SensorType = SensorType.MICROBOLOMETER,
        resolution: tuple[int, int] = (640, 480),
        manufacturer: str = "",
        model: str = "",
    ):
        super().__init__(device_id, device_name, manufacturer, model)
        self.sensor_type = sensor_type
        self.resolution = resolution

        self._sensor = self._create_sensor()
        self._emissivity: float = 0.98  # Human skin emissivity
        self._ambient_temperature: float = 22.0  # Celsius
        self._distance: float = 1.0  # meters
        self._atmospheric_transmission: float = 0.95

        self._nuc_table: Optional[np.ndarray] = None  # Non-Uniformity Correction
        self._bad_pixel_map: Optional[np.ndarray] = None

    def _create_sensor(self) -> ThermalSensor:
        """Create thermal sensor based on type."""
        if self.sensor_type == SensorType.MICROBOLOMETER:
            return ThermalSensor(
                sensor_type=SensorType.MICROBOLOMETER,
                resolution=self.resolution,
                pixel_pitch=17.0,
                netd=50.0,  # mK
                spectral_range=(7.5, 14.0),  # LWIR
                frame_rate=30.0,
            )
        elif self.sensor_type == SensorType.PHOTON_DETECTOR:
            return ThermalSensor(
                sensor_type=SensorType.PHOTON_DETECTOR,
                resolution=self.resolution,
                pixel_pitch=15.0,
                netd=20.0,
                spectral_range=(3.0, 5.0),  # MWIR
                frame_rate=60.0,
                fpa_temperature=77.0,  # Cooled to liquid nitrogen temp
            )
        else:
            return ThermalSensor(
                sensor_type=self.sensor_type,
                resolution=self.resolution,
                pixel_pitch=17.0,
                netd=50.0,
                spectral_range=(7.5, 14.0),
                frame_rate=30.0,
            )

    def initialize(self) -> bool:
        """Initialize the thermographic camera."""
        try:
            self._status = DeviceStatus.INITIALIZING

            # Sensor warm-up / cooling
            self._stabilize_sensor()

            # Load calibration data
            self._load_calibration_data()

            # Perform NUC
            self._perform_nuc()

            self._status = DeviceStatus.READY
            return True
        except Exception as e:
            self.log_error(f"Initialization failed: {str(e)}")
            return False

    def _stabilize_sensor(self):
        """Stabilize sensor temperature."""
        # For cooled sensors, wait for cooling
        # For uncooled sensors, wait for thermal equilibrium
        pass

    def _load_calibration_data(self):
        """Load factory calibration data."""
        # Load radiometric calibration coefficients
        pass

    def _perform_nuc(self):
        """Perform Non-Uniformity Correction."""
        # Generate simulated NUC table
        self._nuc_table = np.ones(self.resolution) + np.random.randn(*self.resolution) * 0.01
        self._bad_pixel_map = np.random.random(self.resolution) > 0.9999

    def shutdown(self) -> bool:
        """Shutdown the camera safely."""
        try:
            self._status = DeviceStatus.OFFLINE
            return True
        except Exception as e:
            self.log_error(f"Shutdown failed: {str(e)}")
            return False

    def acquire(
        self,
        emissivity: Optional[float] = None,
        distance: Optional[float] = None,
        ambient_temp: Optional[float] = None,
        **params,
    ) -> ImageData:
        """Acquire thermal image."""
        if not self.is_ready:
            raise RuntimeError(f"Device not ready. Current status: {self._status}")

        if emissivity is not None:
            self._emissivity = emissivity
        if distance is not None:
            self._distance = distance
        if ambient_temp is not None:
            self._ambient_temperature = ambient_temp

        self._status = DeviceStatus.ACQUIRING

        try:
            # Acquire raw thermal data
            raw_data = self._acquire_raw_frame()

            # Apply corrections
            corrected_data = self._apply_corrections(raw_data)

            # Convert to temperature
            temperature_map = self._convert_to_temperature(corrected_data)

            self._last_acquisition = datetime.now()
            self._status = DeviceStatus.READY

            return ImageData(
                data=temperature_map,
                timestamp=self._last_acquisition,
                metadata={
                    "modality": "thermal",
                    "sensor_type": self.sensor_type.value,
                    "emissivity": self._emissivity,
                    "distance_m": self._distance,
                    "ambient_temp_c": self._ambient_temperature,
                    "min_temp_c": float(temperature_map.min()),
                    "max_temp_c": float(temperature_map.max()),
                    "mean_temp_c": float(temperature_map.mean()),
                    "resolution": self.resolution,
                    "netd_mk": self._sensor.netd,
                    "spectral_range_um": self._sensor.spectral_range,
                },
                device_id=self.device_id,
                modality="thermal",
            )
        except Exception as e:
            self.log_error(f"Acquisition failed: {str(e)}")
            raise

    def _acquire_raw_frame(self) -> np.ndarray:
        """Acquire raw sensor data."""
        # Simulate thermal sensor output
        # Base temperature distribution (body with some variation)
        base_temp = 33.0  # Surface skin temperature

        # Create realistic body temperature pattern
        y, x = np.ogrid[: self.resolution[0], : self.resolution[1]]
        center_y, center_x = self.resolution[0] // 2, self.resolution[1] // 2

        # Distance from center
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = np.sqrt(center_x**2 + center_y**2)

        # Temperature decreases toward edges (body shape approximation)
        temp_pattern = base_temp + 2 * (1 - dist / max_dist)

        # Add some anatomical variations (warmer areas)
        # Face region
        face_mask = dist < max_dist * 0.3
        temp_pattern[face_mask] += 1.5

        # Add noise based on NETD
        noise = np.random.randn(*self.resolution) * (self._sensor.netd / 1000)
        raw_data = temp_pattern + noise

        # Convert to raw counts (simulating 14-bit sensor)
        raw_counts = ((raw_data - 20) / 30 * 16383).astype(np.uint16)

        return raw_counts

    def _apply_corrections(self, raw_data: np.ndarray) -> np.ndarray:
        """Apply sensor corrections."""
        corrected = raw_data.astype(np.float64)

        # Apply NUC
        if self._nuc_table is not None:
            corrected *= self._nuc_table

        # Bad pixel replacement
        if self._bad_pixel_map is not None:
            from scipy.ndimage import median_filter

            median_img = median_filter(corrected, size=3)
            corrected[self._bad_pixel_map] = median_img[self._bad_pixel_map]

        return corrected

    def _convert_to_temperature(self, corrected_data: np.ndarray) -> np.ndarray:
        """Convert corrected sensor data to temperature."""
        # Simplified radiometric conversion
        # In real systems, this uses Planck's law and calibration coefficients

        # Reverse the raw counts to temperature conversion
        temperature = (corrected_data / 16383 * 30) + 20

        # Apply emissivity correction
        # T_object = T_measured / emissivity^0.25 (simplified)
        temperature = temperature / (self._emissivity**0.25)

        # Apply atmospheric correction (simplified)
        temperature = temperature * self._atmospheric_transmission

        return temperature

    def calibrate(
        self,
        blackbody_temps: Optional[list[float]] = None,
        **params,
    ) -> DeviceCalibration:
        """Calibrate the camera against blackbody sources."""
        self._status = DeviceStatus.MAINTENANCE

        try:
            if blackbody_temps is None:
                blackbody_temps = [25.0, 35.0, 45.0]

            calibration_params = {
                "nuc_performed": True,
                "radiometric_calibration": self._perform_radiometric_calibration(blackbody_temps),
                "bad_pixel_count": np.sum(self._bad_pixel_map) if self._bad_pixel_map is not None else 0,
                "accuracy_at_35c": 0.3,  # ±0.3°C
            }

            self._calibration = DeviceCalibration(
                calibration_date=datetime.now(),
                calibration_parameters=calibration_params,
                is_valid=True,
                expiry_date=datetime.now() + timedelta(days=365),
            )

            self._status = DeviceStatus.READY
            return self._calibration
        except Exception as e:
            self.log_error(f"Calibration failed: {str(e)}")
            raise

    def _perform_radiometric_calibration(
        self,
        blackbody_temps: list[float],
    ) -> dict:
        """Perform radiometric calibration using blackbody sources."""
        # Simulated calibration results
        return {
            "calibration_points": blackbody_temps,
            "r_squared": 0.9998,
            "max_error": 0.2,
            "coefficients": [0.001, 20.0, 0.0],  # Simplified linear fit
        }

    def perform_nuc(self, flat_field: Optional[np.ndarray] = None):
        """Perform Non-Uniformity Correction."""
        if flat_field is None:
            # Use internal shutter or uniform source
            self._nuc_table = np.ones(self.resolution)
        else:
            mean_val = flat_field.mean()
            self._nuc_table = mean_val / (flat_field + 1e-10)

    def set_emissivity(self, emissivity: float):
        """Set target emissivity."""
        if not (0.0 < emissivity <= 1.0):
            raise ValueError("Emissivity must be between 0 and 1")
        self._emissivity = emissivity

    def set_distance(self, distance: float):
        """Set distance to target in meters."""
        if distance <= 0:
            raise ValueError("Distance must be positive")
        self._distance = distance

    def set_ambient_temperature(self, temperature: float):
        """Set ambient temperature in Celsius."""
        self._ambient_temperature = temperature

    @property
    def sensor_specs(self) -> ThermalSensor:
        """Get sensor specifications."""
        return self._sensor

    def get_temperature_at_point(
        self,
        image_data: ImageData,
        point: tuple[int, int],
    ) -> float:
        """Get temperature at specific pixel coordinates."""
        x, y = point
        if 0 <= y < image_data.data.shape[0] and 0 <= x < image_data.data.shape[1]:
            return float(image_data.data[y, x])
        raise ValueError("Point outside image bounds")

    def get_roi_statistics(
        self,
        image_data: ImageData,
        roi: tuple[int, int, int, int],  # (x, y, width, height)
    ) -> dict:
        """Get temperature statistics for a region of interest."""
        x, y, w, h = roi
        roi_data = image_data.data[y : y + h, x : x + w]

        return {
            "min": float(roi_data.min()),
            "max": float(roi_data.max()),
            "mean": float(roi_data.mean()),
            "std": float(roi_data.std()),
            "median": float(np.median(roi_data)),
        }
