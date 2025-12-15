"""Inflammation detection algorithms for thermal imaging."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import numpy as np
from scipy import ndimage
from scipy.stats import zscore


class InflammationSeverity(Enum):
    """Inflammation severity levels."""

    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class InflammationRegion:
    """Detected inflammation region."""

    region_id: int
    centroid: tuple[float, float]
    bounding_box: tuple[int, int, int, int]  # (x, y, width, height)
    area_pixels: int
    mean_temperature: float
    max_temperature: float
    temperature_delta: float  # Difference from surrounding tissue
    severity: InflammationSeverity
    confidence: float


@dataclass
class AnalysisResult:
    """Result of inflammation analysis."""

    timestamp: datetime = field(default_factory=datetime.now)
    regions: list[InflammationRegion] = field(default_factory=list)
    overall_assessment: str = ""
    reference_temperature: float = 0.0
    analysis_parameters: dict = field(default_factory=dict)

    @property
    def has_inflammation(self) -> bool:
        """Check if any inflammation was detected."""
        return len(self.regions) > 0

    @property
    def most_severe_region(self) -> Optional[InflammationRegion]:
        """Get the most severe inflammation region."""
        if not self.regions:
            return None
        severity_order = {
            InflammationSeverity.SEVERE: 3,
            InflammationSeverity.MODERATE: 2,
            InflammationSeverity.MILD: 1,
            InflammationSeverity.NONE: 0,
        }
        return max(self.regions, key=lambda r: severity_order[r.severity])


class InflammationDetector:
    """Detector for inflammation in thermal images."""

    def __init__(
        self,
        temperature_threshold: float = 1.5,  # Celsius above reference
        min_region_size: int = 100,  # Minimum pixels
        reference_method: str = "contralateral",
    ):
        self.temperature_threshold = temperature_threshold
        self.min_region_size = min_region_size
        self.reference_method = reference_method

        self._severity_thresholds = {
            "mild": 1.0,  # 1.0-1.5°C above reference
            "moderate": 1.5,  # 1.5-2.5°C above reference
            "severe": 2.5,  # >2.5°C above reference
        }

    def analyze(
        self,
        thermal_image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        reference_region: Optional[tuple[int, int, int, int]] = None,
    ) -> AnalysisResult:
        """Analyze thermal image for inflammation."""
        # Apply mask if provided
        if mask is not None:
            analysis_region = thermal_image.copy()
            analysis_region[~mask] = np.nan
        else:
            analysis_region = thermal_image

        # Determine reference temperature
        reference_temp = self._calculate_reference_temperature(
            analysis_region, reference_region
        )

        # Detect hot spots
        hot_regions = self._detect_hot_regions(analysis_region, reference_temp)

        # Analyze each region
        inflammation_regions = []
        for region_id, region_mask in enumerate(hot_regions):
            region_info = self._analyze_region(
                thermal_image, region_mask, reference_temp, region_id
            )
            if region_info is not None:
                inflammation_regions.append(region_info)

        # Generate overall assessment
        assessment = self._generate_assessment(inflammation_regions)

        return AnalysisResult(
            regions=inflammation_regions,
            overall_assessment=assessment,
            reference_temperature=reference_temp,
            analysis_parameters={
                "temperature_threshold": self.temperature_threshold,
                "min_region_size": self.min_region_size,
                "reference_method": self.reference_method,
            },
        )

    def _calculate_reference_temperature(
        self,
        image: np.ndarray,
        reference_region: Optional[tuple[int, int, int, int]] = None,
    ) -> float:
        """Calculate reference temperature for comparison."""
        if reference_region is not None:
            x, y, w, h = reference_region
            roi = image[y : y + h, x : x + w]
            return float(np.nanmean(roi))

        if self.reference_method == "contralateral":
            # Use contralateral (opposite side) as reference
            # Assume symmetric anatomy
            h, w = image.shape
            left_half = image[:, : w // 2]
            right_half = image[:, w // 2 :]
            return float(np.nanmean([np.nanmean(left_half), np.nanmean(right_half)]))

        elif self.reference_method == "percentile":
            # Use lower percentile as reference (healthy tissue)
            return float(np.nanpercentile(image, 25))

        elif self.reference_method == "global_mean":
            return float(np.nanmean(image))

        else:
            return float(np.nanmean(image))

    def _detect_hot_regions(
        self,
        image: np.ndarray,
        reference_temp: float,
    ) -> list[np.ndarray]:
        """Detect regions with elevated temperature."""
        # Create temperature difference map
        temp_diff = image - reference_temp

        # Threshold to find hot spots
        hot_mask = temp_diff > self.temperature_threshold

        # Remove small regions
        hot_mask = ndimage.binary_opening(hot_mask, iterations=2)

        # Label connected regions
        labeled_array, num_features = ndimage.label(hot_mask)

        # Extract individual region masks
        regions = []
        for i in range(1, num_features + 1):
            region_mask = labeled_array == i
            if np.sum(region_mask) >= self.min_region_size:
                regions.append(region_mask)

        return regions

    def _analyze_region(
        self,
        image: np.ndarray,
        region_mask: np.ndarray,
        reference_temp: float,
        region_id: int,
    ) -> Optional[InflammationRegion]:
        """Analyze a single inflammation region."""
        # Get region temperatures
        region_temps = image[region_mask]

        if len(region_temps) == 0:
            return None

        mean_temp = float(np.mean(region_temps))
        max_temp = float(np.max(region_temps))
        temp_delta = mean_temp - reference_temp

        # Calculate centroid
        y_coords, x_coords = np.where(region_mask)
        centroid = (float(np.mean(x_coords)), float(np.mean(y_coords)))

        # Calculate bounding box
        min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))
        min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))
        bounding_box = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)

        # Determine severity
        severity = self._classify_severity(temp_delta)

        # Calculate confidence based on region characteristics
        confidence = self._calculate_confidence(
            region_mask, region_temps, reference_temp
        )

        return InflammationRegion(
            region_id=region_id,
            centroid=centroid,
            bounding_box=bounding_box,
            area_pixels=int(np.sum(region_mask)),
            mean_temperature=mean_temp,
            max_temperature=max_temp,
            temperature_delta=temp_delta,
            severity=severity,
            confidence=confidence,
        )

    def _classify_severity(self, temp_delta: float) -> InflammationSeverity:
        """Classify inflammation severity based on temperature difference."""
        if temp_delta >= self._severity_thresholds["severe"]:
            return InflammationSeverity.SEVERE
        elif temp_delta >= self._severity_thresholds["moderate"]:
            return InflammationSeverity.MODERATE
        elif temp_delta >= self._severity_thresholds["mild"]:
            return InflammationSeverity.MILD
        else:
            return InflammationSeverity.NONE

    def _calculate_confidence(
        self,
        region_mask: np.ndarray,
        region_temps: np.ndarray,
        reference_temp: float,
    ) -> float:
        """Calculate detection confidence."""
        # Factors affecting confidence:
        # 1. Region size (larger = more confident)
        # 2. Temperature consistency within region
        # 3. Temperature difference from reference

        region_size = np.sum(region_mask)
        size_score = min(region_size / 500, 1.0)  # Max out at 500 pixels

        # Temperature consistency (lower std = more confident)
        temp_std = np.std(region_temps)
        consistency_score = max(0, 1 - temp_std / 2)

        # Temperature difference significance
        temp_diff = np.mean(region_temps) - reference_temp
        diff_score = min(temp_diff / 3, 1.0)  # Max out at 3°C difference

        # Weighted average
        confidence = 0.3 * size_score + 0.3 * consistency_score + 0.4 * diff_score

        return float(np.clip(confidence, 0, 1))

    def _generate_assessment(
        self,
        regions: list[InflammationRegion],
    ) -> str:
        """Generate overall assessment text."""
        if not regions:
            return "No significant inflammation detected."

        severe_count = sum(1 for r in regions if r.severity == InflammationSeverity.SEVERE)
        moderate_count = sum(1 for r in regions if r.severity == InflammationSeverity.MODERATE)
        mild_count = sum(1 for r in regions if r.severity == InflammationSeverity.MILD)

        assessment_parts = []

        if severe_count > 0:
            assessment_parts.append(f"{severe_count} severe inflammation region(s)")
        if moderate_count > 0:
            assessment_parts.append(f"{moderate_count} moderate inflammation region(s)")
        if mild_count > 0:
            assessment_parts.append(f"{mild_count} mild inflammation region(s)")

        total = len(regions)
        max_delta = max(r.temperature_delta for r in regions)

        assessment = f"Detected {total} inflammation region(s): {', '.join(assessment_parts)}. "
        assessment += f"Maximum temperature elevation: {max_delta:.1f}°C above reference."

        return assessment

    def compare_bilateral(
        self,
        thermal_image: np.ndarray,
        left_roi: tuple[int, int, int, int],
        right_roi: tuple[int, int, int, int],
    ) -> dict:
        """Compare temperature between bilateral (left/right) regions."""
        x1, y1, w1, h1 = left_roi
        x2, y2, w2, h2 = right_roi

        left_temps = thermal_image[y1 : y1 + h1, x1 : x1 + w1]
        right_temps = thermal_image[y2 : y2 + h2, x2 : x2 + w2]

        left_mean = float(np.mean(left_temps))
        right_mean = float(np.mean(right_temps))
        difference = abs(left_mean - right_mean)

        asymmetry_significant = difference > 0.5  # Clinical threshold

        return {
            "left_mean_temp": left_mean,
            "right_mean_temp": right_mean,
            "temperature_difference": difference,
            "asymmetry_significant": asymmetry_significant,
            "warmer_side": "left" if left_mean > right_mean else "right",
        }

    def track_inflammation(
        self,
        images: list[np.ndarray],
        timestamps: list[datetime],
    ) -> dict:
        """Track inflammation changes over time."""
        results = []

        for image in images:
            analysis = self.analyze(image)
            if analysis.has_inflammation:
                most_severe = analysis.most_severe_region
                results.append(
                    {
                        "max_temp_delta": most_severe.temperature_delta if most_severe else 0,
                        "total_area": sum(r.area_pixels for r in analysis.regions),
                        "region_count": len(analysis.regions),
                    }
                )
            else:
                results.append({"max_temp_delta": 0, "total_area": 0, "region_count": 0})

        # Analyze trend
        temp_deltas = [r["max_temp_delta"] for r in results]

        if len(temp_deltas) >= 2:
            trend = "improving" if temp_deltas[-1] < temp_deltas[0] else "worsening"
            if abs(temp_deltas[-1] - temp_deltas[0]) < 0.2:
                trend = "stable"
        else:
            trend = "insufficient data"

        return {
            "measurements": results,
            "timestamps": timestamps,
            "trend": trend,
            "initial_delta": temp_deltas[0] if temp_deltas else 0,
            "final_delta": temp_deltas[-1] if temp_deltas else 0,
        }

    def set_severity_thresholds(
        self,
        mild: float = 1.0,
        moderate: float = 1.5,
        severe: float = 2.5,
    ):
        """Set custom severity thresholds."""
        self._severity_thresholds = {
            "mild": mild,
            "moderate": moderate,
            "severe": severe,
        }
