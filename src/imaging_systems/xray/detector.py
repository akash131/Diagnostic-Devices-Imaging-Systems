"""X-ray detector implementation."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
import numpy as np


class DetectorType(Enum):
    """Types of X-ray detectors."""

    FLAT_PANEL_INDIRECT = "flat_panel_indirect"
    FLAT_PANEL_DIRECT = "flat_panel_direct"
    CR_PLATE = "cr_plate"
    CCD = "ccd"


@dataclass
class DetectorSpecs:
    """X-ray detector specifications."""

    pixel_pitch: float  # mm
    active_area: tuple[float, float]  # mm (width, height)
    resolution: tuple[int, int]  # pixels
    bit_depth: int
    dqe: float  # Detective Quantum Efficiency at 0 lp/mm
    mtf_at_nyquist: float
    frame_rate: float  # fps


class FlatPanelDetector:
    """Flat panel digital X-ray detector."""

    def __init__(
        self,
        detector_id: str,
        detector_type: DetectorType,
        pixel_pitch: float = 0.15,
        resolution: tuple[int, int] = (2048, 2048),
        bit_depth: int = 14,
    ):
        self.detector_id = detector_id
        self.detector_type = detector_type
        self.pixel_pitch = pixel_pitch
        self.resolution = resolution
        self.bit_depth = bit_depth

        self._specs = DetectorSpecs(
            pixel_pitch=pixel_pitch,
            active_area=(
                pixel_pitch * resolution[0],
                pixel_pitch * resolution[1],
            ),
            resolution=resolution,
            bit_depth=bit_depth,
            dqe=0.65,
            mtf_at_nyquist=0.25,
            frame_rate=30.0,
        )

        self._gain_map: Optional[np.ndarray] = None
        self._offset_map: Optional[np.ndarray] = None
        self._defect_map: Optional[np.ndarray] = None
        self._is_calibrated = False

    def initialize(self) -> bool:
        """Initialize the detector."""
        try:
            # Load calibration maps
            self._load_calibration_maps()
            return True
        except Exception:
            return False

    def _load_calibration_maps(self):
        """Load detector calibration maps."""
        # Generate simulated calibration maps
        self._gain_map = np.ones(self.resolution) + np.random.randn(*self.resolution) * 0.02
        self._offset_map = np.random.randn(*self.resolution) * 10
        self._defect_map = np.random.random(self.resolution) > 0.9999  # ~0.01% defective pixels
        self._is_calibrated = True

    def acquire_frame(self, exposure_time: float = 0.1) -> np.ndarray:
        """Acquire a single frame from the detector."""
        if not self._is_calibrated:
            raise RuntimeError("Detector not calibrated")

        # Simulate raw detector output
        max_value = 2**self.bit_depth - 1

        # Base signal with electronic noise
        raw_frame = np.random.poisson(1000, self.resolution).astype(np.float64)
        raw_frame += np.random.randn(*self.resolution) * 50  # Read noise

        # Apply gain non-uniformity
        if self._gain_map is not None:
            raw_frame *= self._gain_map

        # Add offset
        if self._offset_map is not None:
            raw_frame += self._offset_map

        # Clip to valid range
        raw_frame = np.clip(raw_frame, 0, max_value)

        return raw_frame.astype(np.uint16)

    def apply_corrections(self, raw_frame: np.ndarray) -> np.ndarray:
        """Apply detector corrections to raw frame."""
        corrected = raw_frame.astype(np.float64)

        # Offset correction
        if self._offset_map is not None:
            corrected -= self._offset_map

        # Gain correction
        if self._gain_map is not None:
            corrected /= self._gain_map

        # Defect correction (interpolation)
        if self._defect_map is not None:
            corrected = self._interpolate_defects(corrected)

        return corrected

    def _interpolate_defects(self, image: np.ndarray) -> np.ndarray:
        """Interpolate defective pixels."""
        if self._defect_map is None:
            return image

        from scipy.ndimage import median_filter

        # Simple median interpolation for defective pixels
        median_img = median_filter(image, size=3)
        result = image.copy()
        result[self._defect_map] = median_img[self._defect_map]
        return result

    def calibrate_offset(self, num_frames: int = 10) -> np.ndarray:
        """Calibrate offset map (dark field)."""
        frames = []
        for _ in range(num_frames):
            # Acquire dark frames (no X-ray)
            dark_frame = np.random.randn(*self.resolution) * 50
            frames.append(dark_frame)

        self._offset_map = np.mean(frames, axis=0)
        return self._offset_map

    def calibrate_gain(self, flat_frames: list[np.ndarray]) -> np.ndarray:
        """Calibrate gain map (flat field)."""
        if len(flat_frames) == 0:
            raise ValueError("No flat frames provided")

        # Average flat frames
        avg_flat = np.mean(flat_frames, axis=0)

        # Subtract offset
        if self._offset_map is not None:
            avg_flat -= self._offset_map

        # Calculate gain map (normalize to mean)
        mean_value = avg_flat.mean()
        self._gain_map = avg_flat / mean_value

        return self._gain_map

    def detect_defects(self, threshold: float = 3.0) -> np.ndarray:
        """Detect defective pixels."""
        if self._gain_map is None:
            raise RuntimeError("Gain calibration required first")

        # Find pixels with abnormal gain
        mean_gain = self._gain_map.mean()
        std_gain = self._gain_map.std()

        self._defect_map = np.abs(self._gain_map - mean_gain) > threshold * std_gain

        return self._defect_map

    @property
    def specs(self) -> DetectorSpecs:
        """Get detector specifications."""
        return self._specs

    @property
    def is_calibrated(self) -> bool:
        """Check if detector is calibrated."""
        return self._is_calibrated


class XRayDetector:
    """Generic X-ray detector interface."""

    def __init__(
        self,
        detector_id: str,
        detector_type: DetectorType = DetectorType.FLAT_PANEL_INDIRECT,
    ):
        self.detector_id = detector_id
        self.detector_type = detector_type
        self._panel = FlatPanelDetector(detector_id, detector_type)

    def initialize(self) -> bool:
        """Initialize the detector."""
        return self._panel.initialize()

    def acquire(self, exposure_params: dict = None) -> np.ndarray:
        """Acquire corrected image from detector."""
        raw = self._panel.acquire_frame()
        return self._panel.apply_corrections(raw)

    def calibrate(self) -> bool:
        """Perform full detector calibration."""
        try:
            # Offset calibration
            self._panel.calibrate_offset()

            # Generate simulated flat frames for gain calibration
            flat_frames = [
                np.random.poisson(5000, self._panel.resolution).astype(np.float64)
                for _ in range(5)
            ]
            self._panel.calibrate_gain(flat_frames)

            # Defect detection
            self._panel.detect_defects()

            return True
        except Exception:
            return False

    @property
    def resolution(self) -> tuple[int, int]:
        """Get detector resolution."""
        return self._panel.resolution

    @property
    def pixel_size(self) -> float:
        """Get pixel size in mm."""
        return self._panel.pixel_pitch
