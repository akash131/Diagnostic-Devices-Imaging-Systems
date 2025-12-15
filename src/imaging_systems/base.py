"""Base classes for diagnostic imaging devices."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import numpy as np


class DeviceStatus(Enum):
    """Device operational status."""

    OFFLINE = "offline"
    INITIALIZING = "initializing"
    READY = "ready"
    ACQUIRING = "acquiring"
    PROCESSING = "processing"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class ImageData:
    """Container for imaging data."""

    data: np.ndarray
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    device_id: str = ""
    modality: str = ""

    @property
    def shape(self) -> tuple:
        """Return the shape of the image data."""
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        """Return the data type of the image."""
        return self.data.dtype

    def normalize(self) -> "ImageData":
        """Return a normalized copy of the image data."""
        normalized = self.data.astype(np.float64)
        min_val, max_val = normalized.min(), normalized.max()
        if max_val > min_val:
            normalized = (normalized - min_val) / (max_val - min_val)
        return ImageData(
            data=normalized,
            timestamp=self.timestamp,
            metadata={**self.metadata, "normalized": True},
            device_id=self.device_id,
            modality=self.modality,
        )


@dataclass
class DeviceCalibration:
    """Calibration data for imaging devices."""

    calibration_date: datetime
    calibration_parameters: dict
    is_valid: bool = True
    expiry_date: Optional[datetime] = None

    def check_validity(self) -> bool:
        """Check if calibration is still valid."""
        if not self.is_valid:
            return False
        if self.expiry_date and datetime.now() > self.expiry_date:
            return False
        return True


class ImagingDevice(ABC):
    """Abstract base class for all imaging devices."""

    def __init__(
        self,
        device_id: str,
        device_name: str,
        manufacturer: str = "",
        model: str = "",
    ):
        self.device_id = device_id
        self.device_name = device_name
        self.manufacturer = manufacturer
        self.model = model
        self._status = DeviceStatus.OFFLINE
        self._calibration: Optional[DeviceCalibration] = None
        self._last_acquisition: Optional[datetime] = None
        self._error_log: list[str] = []

    @property
    def status(self) -> DeviceStatus:
        """Get current device status."""
        return self._status

    @status.setter
    def status(self, value: DeviceStatus):
        """Set device status."""
        self._status = value

    @property
    def is_ready(self) -> bool:
        """Check if device is ready for acquisition."""
        return self._status == DeviceStatus.READY

    @property
    def calibration(self) -> Optional[DeviceCalibration]:
        """Get current calibration data."""
        return self._calibration

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the device."""
        pass

    @abstractmethod
    def shutdown(self) -> bool:
        """Shutdown the device safely."""
        pass

    @abstractmethod
    def acquire(self, **params) -> ImageData:
        """Acquire image data from the device."""
        pass

    @abstractmethod
    def calibrate(self, **params) -> DeviceCalibration:
        """Calibrate the device."""
        pass

    def get_device_info(self) -> dict[str, Any]:
        """Get device information."""
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "status": self._status.value,
            "calibrated": self._calibration is not None and self._calibration.check_validity(),
            "last_acquisition": self._last_acquisition.isoformat() if self._last_acquisition else None,
        }

    def log_error(self, error_message: str):
        """Log an error message."""
        timestamp = datetime.now().isoformat()
        self._error_log.append(f"[{timestamp}] {error_message}")
        self._status = DeviceStatus.ERROR

    def clear_errors(self):
        """Clear error log and reset status."""
        self._error_log.clear()
        self._status = DeviceStatus.READY


class SignalProcessorBase(ABC):
    """Base class for signal processing units."""

    def __init__(self, processor_id: str, sampling_rate: float):
        self.processor_id = processor_id
        self.sampling_rate = sampling_rate
        self._filters: list[callable] = []

    @abstractmethod
    def process(self, signal: np.ndarray) -> np.ndarray:
        """Process input signal."""
        pass

    def add_filter(self, filter_func: callable):
        """Add a filter to the processing chain."""
        self._filters.append(filter_func)

    def clear_filters(self):
        """Clear all filters."""
        self._filters.clear()

    def apply_filters(self, signal: np.ndarray) -> np.ndarray:
        """Apply all filters in sequence."""
        result = signal.copy()
        for filter_func in self._filters:
            result = filter_func(result)
        return result
