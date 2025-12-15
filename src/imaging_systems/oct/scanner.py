"""OCT scanner and galvanometer control implementation."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable
import numpy as np


class ScanMode(Enum):
    """Scanner operating modes."""

    BIDIRECTIONAL = "bidirectional"
    UNIDIRECTIONAL = "unidirectional"
    FLYBACK = "flyback"


@dataclass
class ScanParameters:
    """Parameters for OCT scanning."""

    scan_width: float  # mm
    scan_height: float  # mm (for volume scans)
    num_a_scans: int
    num_b_scans: int
    scan_rate: float  # Hz
    flyback_time: float  # ms
    bidirectional: bool = False


@dataclass
class GalvoLimits:
    """Galvanometer position limits."""

    x_min: float  # V
    x_max: float  # V
    y_min: float  # V
    y_max: float  # V
    max_velocity: float  # V/ms


class GalvoScanner:
    """Galvanometer scanner controller."""

    def __init__(
        self,
        x_sensitivity: float = 1.0,  # mm/V
        y_sensitivity: float = 1.0,  # mm/V
    ):
        self.x_sensitivity = x_sensitivity
        self.y_sensitivity = y_sensitivity

        self._limits = GalvoLimits(
            x_min=-10.0,
            x_max=10.0,
            y_min=-10.0,
            y_max=10.0,
            max_velocity=50.0,
        )

        self._current_x: float = 0.0
        self._current_y: float = 0.0
        self._is_initialized = False

    def initialize(self) -> bool:
        """Initialize galvanometer scanners."""
        try:
            # Move to home position
            self._current_x = 0.0
            self._current_y = 0.0
            self._is_initialized = True
            return True
        except Exception:
            return False

    def generate_scan_pattern(
        self,
        params: ScanParameters,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate voltage waveforms for scan pattern."""
        # Convert scan dimensions to voltages
        x_amplitude = params.scan_width / (2 * self.x_sensitivity)
        y_amplitude = params.scan_height / (2 * self.y_sensitivity)

        # Check limits
        if abs(x_amplitude) > self._limits.x_max:
            raise ValueError(f"X scan exceeds galvo limits: {x_amplitude}V > {self._limits.x_max}V")
        if abs(y_amplitude) > self._limits.y_max:
            raise ValueError(f"Y scan exceeds galvo limits: {y_amplitude}V > {self._limits.y_max}V")

        # Generate X waveform (fast scan)
        samples_per_b_scan = params.num_a_scans
        flyback_samples = int(params.flyback_time * params.scan_rate / 1000)

        if params.bidirectional:
            x_wave = self._generate_bidirectional_wave(
                x_amplitude, samples_per_b_scan
            )
        else:
            x_wave = self._generate_sawtooth_wave(
                x_amplitude, samples_per_b_scan, flyback_samples
            )

        # Repeat for all B-scans
        x_waveform = np.tile(x_wave, params.num_b_scans)

        # Generate Y waveform (slow scan)
        total_samples = len(x_waveform)
        y_waveform = np.linspace(-y_amplitude, y_amplitude, total_samples)

        return x_waveform, y_waveform

    def _generate_sawtooth_wave(
        self,
        amplitude: float,
        samples: int,
        flyback_samples: int,
    ) -> np.ndarray:
        """Generate sawtooth waveform with flyback."""
        forward = np.linspace(-amplitude, amplitude, samples)
        flyback = np.linspace(amplitude, -amplitude, flyback_samples)
        return np.concatenate([forward, flyback])

    def _generate_bidirectional_wave(
        self,
        amplitude: float,
        samples: int,
    ) -> np.ndarray:
        """Generate bidirectional triangle waveform."""
        forward = np.linspace(-amplitude, amplitude, samples)
        backward = np.linspace(amplitude, -amplitude, samples)
        return np.concatenate([forward, backward])

    def generate_radial_pattern(
        self,
        scan_diameter: float,
        num_radials: int,
        samples_per_radial: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate radial (star) scan pattern."""
        amplitude = scan_diameter / (2 * self.x_sensitivity)

        angles = np.linspace(0, np.pi, num_radials, endpoint=False)
        x_waveform = []
        y_waveform = []

        for angle in angles:
            # Each radial goes from -R to +R through center
            t = np.linspace(-1, 1, samples_per_radial)
            x_waveform.extend(amplitude * t * np.cos(angle))
            y_waveform.extend(amplitude * t * np.sin(angle))

        return np.array(x_waveform), np.array(y_waveform)

    def generate_circular_pattern(
        self,
        diameter: float,
        num_points: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate circular scan pattern."""
        radius = diameter / (2 * self.x_sensitivity)
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        x_waveform = radius * np.cos(angles)
        y_waveform = radius * np.sin(angles)

        return x_waveform, y_waveform

    def set_position(self, x: float, y: float):
        """Set scanner to specific position."""
        if not self._is_initialized:
            raise RuntimeError("Scanner not initialized")

        x_voltage = x / self.x_sensitivity
        y_voltage = y / self.y_sensitivity

        if not (self._limits.x_min <= x_voltage <= self._limits.x_max):
            raise ValueError(f"X position out of range: {x_voltage}V")
        if not (self._limits.y_min <= y_voltage <= self._limits.y_max):
            raise ValueError(f"Y position out of range: {y_voltage}V")

        self._current_x = x_voltage
        self._current_y = y_voltage

    @property
    def current_position(self) -> tuple[float, float]:
        """Get current scanner position in mm."""
        return (
            self._current_x * self.x_sensitivity,
            self._current_y * self.y_sensitivity,
        )


class OCTScanner:
    """High-level OCT scanner interface."""

    def __init__(
        self,
        galvo_x_sensitivity: float = 1.0,
        galvo_y_sensitivity: float = 1.0,
    ):
        self._galvo = GalvoScanner(galvo_x_sensitivity, galvo_y_sensitivity)
        self._current_params: Optional[ScanParameters] = None
        self._scan_callback: Optional[Callable] = None

    def initialize(self) -> bool:
        """Initialize the scanner system."""
        return self._galvo.initialize()

    def configure_scan(self, params: ScanParameters):
        """Configure scan parameters."""
        self._current_params = params

    def generate_waveforms(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate scan waveforms based on current configuration."""
        if self._current_params is None:
            raise RuntimeError("Scan not configured")

        return self._galvo.generate_scan_pattern(self._current_params)

    def start_scan(self, callback: Optional[Callable] = None):
        """Start scanning with optional callback for each A-scan."""
        if self._current_params is None:
            raise RuntimeError("Scan not configured")

        self._scan_callback = callback
        x_wave, y_wave = self.generate_waveforms()

        # In real implementation, this would drive the galvos
        # and trigger data acquisition
        return x_wave, y_wave

    def stop_scan(self):
        """Stop current scan."""
        self._galvo.set_position(0, 0)

    def move_to(self, x: float, y: float):
        """Move scanner to specific position."""
        self._galvo.set_position(x, y)

    def set_scan_pattern(
        self,
        pattern_type: str,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Set predefined scan pattern."""
        if pattern_type == "linear":
            params = ScanParameters(
                scan_width=kwargs.get("width", 6.0),
                scan_height=0,
                num_a_scans=kwargs.get("num_a_scans", 512),
                num_b_scans=1,
                scan_rate=kwargs.get("scan_rate", 70000),
                flyback_time=kwargs.get("flyback_time", 0.5),
            )
            self._current_params = params
            return self._galvo.generate_scan_pattern(params)

        elif pattern_type == "radial":
            return self._galvo.generate_radial_pattern(
                scan_diameter=kwargs.get("diameter", 6.0),
                num_radials=kwargs.get("num_radials", 12),
                samples_per_radial=kwargs.get("samples", 512),
            )

        elif pattern_type == "circular":
            return self._galvo.generate_circular_pattern(
                diameter=kwargs.get("diameter", 3.4),
                num_points=kwargs.get("num_points", 768),
            )

        elif pattern_type == "volume":
            params = ScanParameters(
                scan_width=kwargs.get("width", 6.0),
                scan_height=kwargs.get("height", 6.0),
                num_a_scans=kwargs.get("num_a_scans", 512),
                num_b_scans=kwargs.get("num_b_scans", 128),
                scan_rate=kwargs.get("scan_rate", 70000),
                flyback_time=kwargs.get("flyback_time", 0.5),
            )
            self._current_params = params
            return self._galvo.generate_scan_pattern(params)

        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")

    @property
    def current_position(self) -> tuple[float, float]:
        """Get current scanner position."""
        return self._galvo.current_position
