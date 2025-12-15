"""Phantom analysis for quality assurance."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import numpy as np
from scipy import ndimage


class PhantomType(Enum):
    """Types of QA phantoms."""

    # Ultrasound
    TISSUE_MIMICKING = "tissue_mimicking"
    DOPPLER = "doppler"
    RESOLUTION = "resolution"

    # X-ray
    LINE_PAIR = "line_pair"
    CONTRAST_DETAIL = "contrast_detail"
    UNIFORMITY = "uniformity"
    LEEDS_TOR = "leeds_tor"

    # OCT
    MODEL_EYE = "model_eye"
    LAYER_PHANTOM = "layer_phantom"

    # Thermal
    BLACKBODY = "blackbody"
    DIFFERENTIAL_BLACKBODY = "differential_blackbody"


@dataclass
class PhantomResult:
    """Result of phantom analysis."""

    phantom_type: PhantomType
    passed: bool
    metrics: dict = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "phantom_type": self.phantom_type.value,
            "passed": self.passed,
            "metrics": self.metrics,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
        }


class PhantomAnalyzer:
    """Analyze phantom images for QA purposes."""

    def __init__(self, modality: str):
        self.modality = modality

    def analyze(
        self,
        image: np.ndarray,
        phantom_type: PhantomType,
        expected_values: dict = None,
    ) -> PhantomResult:
        """Analyze phantom image."""
        expected = expected_values or {}

        if phantom_type == PhantomType.LINE_PAIR:
            return self._analyze_line_pair(image, expected)
        elif phantom_type == PhantomType.UNIFORMITY:
            return self._analyze_uniformity(image, expected)
        elif phantom_type == PhantomType.CONTRAST_DETAIL:
            return self._analyze_contrast_detail(image, expected)
        elif phantom_type == PhantomType.TISSUE_MIMICKING:
            return self._analyze_tissue_mimicking(image, expected)
        elif phantom_type == PhantomType.BLACKBODY:
            return self._analyze_blackbody(image, expected)
        elif phantom_type == PhantomType.LAYER_PHANTOM:
            return self._analyze_layer_phantom(image, expected)
        else:
            return PhantomResult(
                phantom_type=phantom_type,
                passed=False,
                issues=["Unknown phantom type"],
            )

    def _analyze_line_pair(
        self,
        image: np.ndarray,
        expected: dict,
    ) -> PhantomResult:
        """Analyze line pair phantom for spatial resolution."""
        metrics = {}
        issues = []

        # Find line patterns using FFT
        fft = np.fft.fft2(image)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)

        # Find peak frequencies
        center = np.array(magnitude.shape) // 2
        radial_profile = self._radial_profile(magnitude, center)

        # Estimate resolution
        peaks = self._find_peaks(radial_profile)
        if len(peaks) > 0:
            # Convert to lp/mm based on image size
            highest_freq = peaks[-1]
            resolution_lpmm = highest_freq / (image.shape[0] * 0.1)  # Assuming 0.1mm pixels
            metrics["resolution_lpmm"] = resolution_lpmm

        # Check against expected
        passed = True
        min_resolution = expected.get("min_resolution_lpmm", 2.0)
        if metrics.get("resolution_lpmm", 0) < min_resolution:
            passed = False
            issues.append(f"Resolution {metrics.get('resolution_lpmm', 0):.1f} lp/mm below minimum {min_resolution}")

        return PhantomResult(
            phantom_type=PhantomType.LINE_PAIR,
            passed=passed,
            metrics=metrics,
            issues=issues,
        )

    def _analyze_uniformity(
        self,
        image: np.ndarray,
        expected: dict,
    ) -> PhantomResult:
        """Analyze uniformity phantom."""
        metrics = {}
        issues = []
        recommendations = []

        # Calculate ROI statistics
        h, w = image.shape[:2]
        roi_size = min(h, w) // 8

        # Center ROI
        center_roi = image[
            h // 2 - roi_size:h // 2 + roi_size,
            w // 2 - roi_size:w // 2 + roi_size
        ]

        # Peripheral ROIs
        peripheral_means = []
        positions = [
            (roi_size, roi_size),  # Top-left
            (roi_size, w - roi_size),  # Top-right
            (h - roi_size, roi_size),  # Bottom-left
            (h - roi_size, w - roi_size),  # Bottom-right
        ]

        for y, x in positions:
            roi = image[y - roi_size:y + roi_size, x - roi_size:x + roi_size]
            peripheral_means.append(roi.mean())

        center_mean = center_roi.mean()
        metrics["center_mean"] = float(center_mean)
        metrics["peripheral_means"] = [float(m) for m in peripheral_means]

        # Calculate uniformity indices
        all_means = [center_mean] + peripheral_means
        integral_uniformity = (max(all_means) - min(all_means)) / (max(all_means) + min(all_means))
        metrics["integral_uniformity"] = float(1 - integral_uniformity)

        # Center-to-edge ratio
        edge_mean = np.mean(peripheral_means)
        metrics["center_to_edge_ratio"] = float(center_mean / (edge_mean + 1e-10))

        # Check thresholds
        passed = True
        min_uniformity = expected.get("min_uniformity", 0.80)
        if metrics["integral_uniformity"] < min_uniformity:
            passed = False
            issues.append(f"Uniformity {metrics['integral_uniformity']:.2%} below threshold {min_uniformity:.2%}")
            recommendations.append("Consider recalibrating detector")

        return PhantomResult(
            phantom_type=PhantomType.UNIFORMITY,
            passed=passed,
            metrics=metrics,
            issues=issues,
            recommendations=recommendations,
        )

    def _analyze_contrast_detail(
        self,
        image: np.ndarray,
        expected: dict,
    ) -> PhantomResult:
        """Analyze contrast-detail phantom."""
        metrics = {}
        issues = []

        # Simplified contrast-detail analysis
        # Look for circular targets of varying size and contrast

        # Calculate local contrast
        blurred = ndimage.gaussian_filter(image.astype(float), sigma=5)
        local_contrast = np.abs(image.astype(float) - blurred)

        metrics["mean_local_contrast"] = float(local_contrast.mean())
        metrics["max_local_contrast"] = float(local_contrast.max())

        # Estimate detectable contrast
        noise_std = image.std()
        metrics["noise_std"] = float(noise_std)
        metrics["estimated_detectability"] = float(metrics["mean_local_contrast"] / (noise_std + 1e-10))

        passed = metrics["estimated_detectability"] > expected.get("min_detectability", 3.0)

        return PhantomResult(
            phantom_type=PhantomType.CONTRAST_DETAIL,
            passed=passed,
            metrics=metrics,
            issues=issues,
        )

    def _analyze_tissue_mimicking(
        self,
        image: np.ndarray,
        expected: dict,
    ) -> PhantomResult:
        """Analyze ultrasound tissue-mimicking phantom."""
        metrics = {}
        issues = []

        # Analyze depth penetration
        depth_profile = image.mean(axis=1)
        metrics["mean_intensity"] = float(image.mean())

        # Find depth where signal drops to 50%
        max_intensity = depth_profile.max()
        half_max = max_intensity / 2
        depth_50 = np.argmax(depth_profile < half_max) if (depth_profile < half_max).any() else len(depth_profile)
        metrics["penetration_depth_index"] = int(depth_50)

        # Analyze uniformity in near field
        near_field = image[:len(image) // 3, :]
        metrics["near_field_uniformity"] = float(1 - near_field.std() / (near_field.mean() + 1e-10))

        passed = metrics["near_field_uniformity"] > expected.get("min_uniformity", 0.7)

        return PhantomResult(
            phantom_type=PhantomType.TISSUE_MIMICKING,
            passed=passed,
            metrics=metrics,
            issues=issues,
        )

    def _analyze_blackbody(
        self,
        image: np.ndarray,
        expected: dict,
    ) -> PhantomResult:
        """Analyze thermal blackbody phantom."""
        metrics = {}
        issues = []

        expected_temp = expected.get("temperature", 35.0)

        # Assume image contains temperature values
        measured_temp = image.mean()
        temp_std = image.std()

        metrics["measured_temperature"] = float(measured_temp)
        metrics["temperature_std"] = float(temp_std)
        metrics["temperature_error"] = float(abs(measured_temp - expected_temp))

        # Check accuracy
        max_error = expected.get("max_error", 0.5)  # °C
        passed = metrics["temperature_error"] <= max_error

        if not passed:
            issues.append(f"Temperature error {metrics['temperature_error']:.2f}°C exceeds {max_error}°C")

        return PhantomResult(
            phantom_type=PhantomType.BLACKBODY,
            passed=passed,
            metrics=metrics,
            issues=issues,
        )

    def _analyze_layer_phantom(
        self,
        image: np.ndarray,
        expected: dict,
    ) -> PhantomResult:
        """Analyze OCT layer phantom."""
        metrics = {}
        issues = []

        # Find layer boundaries using edge detection
        edges = ndimage.sobel(image.astype(float), axis=0)

        # Find prominent horizontal edges (layers)
        edge_profile = np.abs(edges).mean(axis=1)
        peaks = self._find_peaks(edge_profile, min_distance=20)

        metrics["detected_layers"] = len(peaks)
        metrics["layer_positions"] = [int(p) for p in peaks]

        # Calculate layer spacing
        if len(peaks) > 1:
            spacings = np.diff(peaks)
            metrics["mean_layer_spacing"] = float(spacings.mean())
            metrics["layer_spacing_std"] = float(spacings.std())

        expected_layers = expected.get("num_layers", 3)
        passed = len(peaks) >= expected_layers

        if not passed:
            issues.append(f"Detected {len(peaks)} layers, expected {expected_layers}")

        return PhantomResult(
            phantom_type=PhantomType.LAYER_PHANTOM,
            passed=passed,
            metrics=metrics,
            issues=issues,
        )

    def _radial_profile(self, image: np.ndarray, center: tuple) -> np.ndarray:
        """Calculate radial profile from center."""
        y, x = np.indices(image.shape)
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
        r = r.astype(int)

        max_r = min(center)
        profile = np.zeros(max_r)
        for i in range(max_r):
            mask = r == i
            if mask.any():
                profile[i] = image[mask].mean()

        return profile

    def _find_peaks(self, data: np.ndarray, min_distance: int = 10) -> np.ndarray:
        """Find peaks in 1D data."""
        peaks = []
        for i in range(min_distance, len(data) - min_distance):
            if data[i] > data[i - 1] and data[i] > data[i + 1]:
                if data[i] > np.mean(data) + np.std(data):
                    peaks.append(i)

        # Filter peaks by minimum distance
        if len(peaks) > 1:
            filtered = [peaks[0]]
            for p in peaks[1:]:
                if p - filtered[-1] >= min_distance:
                    filtered.append(p)
            peaks = filtered

        return np.array(peaks)
