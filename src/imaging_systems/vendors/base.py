"""Base classes for vendor adapters and protocols."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable
import numpy as np

from ..base import ImageData


class ConnectionType(Enum):
    """Connection types for vendor devices."""

    USB = "usb"
    ETHERNET = "ethernet"
    SERIAL = "serial"
    BLUETOOTH = "bluetooth"
    WIFI = "wifi"
    PROPRIETARY = "proprietary"


class DataFormat(Enum):
    """Data formats used by vendors."""

    RAW = "raw"
    DICOM = "dicom"
    PROPRIETARY = "proprietary"
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"


@dataclass
class ConnectionConfig:
    """Configuration for connecting to vendor devices."""

    connection_type: ConnectionType = ConnectionType.ETHERNET
    host: str = ""
    port: int = 0
    username: str = ""
    password: str = ""
    timeout: float = 30.0
    retry_count: int = 3
    ssl_enabled: bool = False
    certificate_path: str = ""
    extra_params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "connection_type": self.connection_type.value,
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "timeout": self.timeout,
            "ssl_enabled": self.ssl_enabled,
        }


@dataclass
class DeviceCapabilities:
    """Capabilities of a vendor device."""

    modalities: list[str]
    supported_formats: list[DataFormat]
    max_resolution: tuple[int, int]
    supports_streaming: bool = False
    supports_remote_control: bool = False
    supports_calibration: bool = True
    supports_dicom_export: bool = True
    max_frame_rate: float = 30.0
    color_depth: int = 8
    features: list[str] = field(default_factory=list)


class VendorProtocol(ABC):
    """Abstract base class for vendor-specific protocols."""

    @abstractmethod
    def connect(self, config: ConnectionConfig) -> bool:
        """Establish connection to device."""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Close connection to device."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to device."""
        pass

    @abstractmethod
    def send_command(self, command: str, params: dict = None) -> dict:
        """Send command to device."""
        pass

    @abstractmethod
    def receive_data(self, timeout: float = None) -> bytes:
        """Receive data from device."""
        pass

    @abstractmethod
    def get_device_info(self) -> dict:
        """Get device information."""
        pass


class VendorAdapter(ABC):
    """Abstract base class for vendor adapters."""

    VENDOR_NAME: str = "Generic"
    VENDOR_ID: str = "generic"
    SUPPORTED_MODELS: list[str] = []

    def __init__(self, model: str = ""):
        self.model = model
        self._protocol: Optional[VendorProtocol] = None
        self._capabilities: Optional[DeviceCapabilities] = None
        self._connected = False
        self._last_error: str = ""
        self._event_handlers: dict[str, list[Callable]] = {}

    @property
    def vendor_name(self) -> str:
        return self.VENDOR_NAME

    @property
    def vendor_id(self) -> str:
        return self.VENDOR_ID

    @abstractmethod
    def get_capabilities(self) -> DeviceCapabilities:
        """Get device capabilities."""
        pass

    @abstractmethod
    def connect(self, config: ConnectionConfig) -> bool:
        """Connect to the device."""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the device."""
        pass

    @abstractmethod
    def acquire_image(self, params: dict = None) -> ImageData:
        """Acquire image from the device."""
        pass

    @abstractmethod
    def configure_device(self, settings: dict) -> bool:
        """Configure device settings."""
        pass

    @abstractmethod
    def get_device_status(self) -> dict:
        """Get current device status."""
        pass

    @abstractmethod
    def export_to_dicom(self, image_data: ImageData) -> bytes:
        """Export image data to DICOM format."""
        pass

    @abstractmethod
    def import_from_vendor_format(self, data: bytes) -> ImageData:
        """Import data from vendor-specific format."""
        pass

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected

    def get_last_error(self) -> str:
        """Get last error message."""
        return self._last_error

    def register_event_handler(self, event: str, handler: Callable):
        """Register event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def _emit_event(self, event: str, data: Any = None):
        """Emit event to registered handlers."""
        if event in self._event_handlers:
            for handler in self._event_handlers[event]:
                try:
                    handler(data)
                except Exception as e:
                    self._last_error = f"Event handler error: {str(e)}"

    def calibrate(self, params: dict = None) -> bool:
        """Calibrate the device (if supported)."""
        if not self._capabilities or not self._capabilities.supports_calibration:
            self._last_error = "Calibration not supported"
            return False
        return True

    def start_streaming(self, callback: Callable[[ImageData], None]) -> bool:
        """Start image streaming (if supported)."""
        if not self._capabilities or not self._capabilities.supports_streaming:
            self._last_error = "Streaming not supported"
            return False
        return True

    def stop_streaming(self) -> bool:
        """Stop image streaming."""
        return True

    def get_supported_settings(self) -> dict:
        """Get supported settings and their ranges."""
        return {}


class SimulatedProtocol(VendorProtocol):
    """Simulated protocol for testing."""

    def __init__(self):
        self._connected = False
        self._device_info = {}

    def connect(self, config: ConnectionConfig) -> bool:
        self._connected = True
        self._device_info = {
            "host": config.host,
            "port": config.port,
            "connected_at": datetime.now().isoformat(),
        }
        return True

    def disconnect(self) -> bool:
        self._connected = False
        return True

    def is_connected(self) -> bool:
        return self._connected

    def send_command(self, command: str, params: dict = None) -> dict:
        return {"status": "ok", "command": command, "params": params}

    def receive_data(self, timeout: float = None) -> bytes:
        return b""

    def get_device_info(self) -> dict:
        return self._device_info
