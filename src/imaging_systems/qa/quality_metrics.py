"""Image quality metrics and assessment."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import numpy as np
from scipy import ndimage


class QualityLevel(Enum):
    """Quality assessment levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


@dataclass
class QualityMetrics:
    """Collection of image quality metrics."""

    # Noise metrics
    snr: float = 0.0  # Signal-to-noise ratio
    cnr: float = 0.0  # Contrast-to-noise ratio

    # Resolution metrics
    mtf50: float = 0.0  # Modulation transfer function at 50%
    spatial_resolution: float = 0.0  # Line pairs per mm

    # Contrast metrics
    contrast: float = 0.0
    dynamic_range: float = 0.0

    # Uniformity metrics
    uniformity: float = 0.0
    non_uniformity_index: float = 0.0

    # Artifact metrics
    artifact_score: float = 0.0
    ghosting_ratio: float = 0.0

    # Overall
    overall_quality: QualityLevel = QualityLevel.ACCEPTABLE
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "snr": self.snr,
            "cnr": self.cnr,
            "mtf50": self.mtf50,
            "spatial_resolution": self.spatial_resolution,
            "contrast": self.contrast,
            "dynamic_range": self.dynamic_range,
            "uniformity": self.uniformity,
            "artifact_score": self.artifact_score,
            "overall_quality": self.overall_quality.value,
        }


class ImageQualityAssessment:
    """Assess image quality for medical imaging."""

    def __init__(self, modality: str = ""):
        self.modality = modality
        self._thresholds = self._get_default_thresholds()

    def _get_default_thresholds(self) -> dict:
        """Get quality thresholds based on modality."""
        return {
            "snr_excellent": 30,
            "snr_good": 20,
            "snr_acceptable": 10,
            "uniformity_excellent": 0.95,
            "uniformity_good": 0.90,
            "uniformity_acceptable": 0.80,
        }

    def assess(self, image: np.ndarray, roi: Optional[np.ndarray] = None) -> QualityMetrics:
        """Perform comprehensive quality assessment."""
        metrics = QualityMetrics()

        # Calculate SNR
        metrics.snr = self._calculate_snr(image, roi)

        # Calculate contrast
        metrics.contrast = self._calculate_contrast(image)

        # Calculate dynamic range
        metrics.dynamic_range = self._calculate_dynamic_range(image)

        # Calculate uniformity
        metrics.uniformity = self._calculate_uniformity(image)

        # Calculate non-uniformity index
        metrics.non_uniformity_index = 1 - metrics.uniformity

        # Detect artifacts
        metrics.artifact_score = self._detect_artifacts(image)

        # Determine overall quality
        metrics.overall_quality = self._determine_overall_quality(metrics)

        return metrics

    def _calculate_snr(
        self,
        image: np.ndarray,
        roi: Optional[np.ndarray] = None,
    ) -> float:
        """Calculate signal-to-noise ratio."""
        if roi is not None:
            signal_region = image[roi]
        else:
            # Use center region as signal
            h, w = image.shape[:2]
            signal_region = image[h // 4:3 * h // 4, w // 4:3 * w // 4]

        signal_mean = signal_region.mean()
        noise_std = signal_region.std()

        if noise_std > 0:
            return signal_mean / noise_std
        return 0.0

    def _calculate_cnr(
        self,
        image: np.ndarray,
        signal_roi: np.ndarray,
        background_roi: np.ndarray,
    ) -> float:
        """Calculate contrast-to-noise ratio."""
        signal = image[signal_roi].mean()
        background = image[background_roi].mean()
        noise = image[background_roi].std()

        if noise > 0:
            return abs(signal - background) / noise
        return 0.0

    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate image contrast."""
        return (image.max() - image.min()) / (image.max() + image.min() + 1e-10)

    def _calculate_dynamic_range(self, image: np.ndarray) -> float:
        """Calculate dynamic range in dB."""
        max_val = image.max()
        min_val = image.min()
        if min_val > 0:
            return 20 * np.log10(max_val / min_val)
        return 0.0

    def _calculate_uniformity(self, image: np.ndarray) -> float:
        """Calculate image uniformity."""
        # Calculate local statistics
        block_size = min(image.shape[:2]) // 8
        means = []

        h, w = image.shape[:2]
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = image[i:i + block_size, j:j + block_size]
                means.append(block.mean())

        if len(means) > 1:
            mean_of_means = np.mean(means)
            std_of_means = np.std(means)
            if mean_of_means > 0:
                return 1 - (std_of_means / mean_of_means)
        return 1.0

    def _detect_artifacts(self, image: np.ndarray) -> float:
        """Detect and score artifacts."""
        # Simple artifact detection based on edge analysis
        edges = ndimage.sobel(image.astype(float))
        edge_std = edges.std()
        image_std = image.std()

        # High edge variance relative to image variance suggests artifacts
        if image_std > 0:
            artifact_ratio = edge_std / image_std
            # Normalize to 0-1 score (lower is better)
            return min(1.0, max(0.0, 1 - artifact_ratio / 10))
        return 0.0

    def _determine_overall_quality(self, metrics: QualityMetrics) -> QualityLevel:
        """Determine overall quality level."""
        scores = []

        # SNR score
        if metrics.snr >= self._thresholds["snr_excellent"]:
            scores.append(4)
        elif metrics.snr >= self._thresholds["snr_good"]:
            scores.append(3)
        elif metrics.snr >= self._thresholds["snr_acceptable"]:
            scores.append(2)
        else:
            scores.append(1)

        # Uniformity score
        if metrics.uniformity >= self._thresholds["uniformity_excellent"]:
            scores.append(4)
        elif metrics.uniformity >= self._thresholds["uniformity_good"]:
            scores.append(3)
        elif metrics.uniformity >= self._thresholds["uniformity_acceptable"]:
            scores.append(2)
        else:
            scores.append(1)

        # Average score
        avg_score = np.mean(scores)

        if avg_score >= 3.5:
            return QualityLevel.EXCELLENT
        elif avg_score >= 2.5:
            return QualityLevel.GOOD
        elif avg_score >= 1.5:
            return QualityLevel.ACCEPTABLE
        elif avg_score >= 1.0:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNACCEPTABLE

    def calculate_mtf(self, edge_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calculate Modulation Transfer Function from edge image."""
        # Find edge and calculate ESF
        profile = edge_image.mean(axis=0)

        # Differentiate to get LSF
        lsf = np.diff(profile)

        # FFT to get MTF
        mtf = np.abs(np.fft.fft(lsf))
        mtf = mtf / mtf[0]  # Normalize

        # Frequency axis
        freq = np.fft.fftfreq(len(lsf))

        return freq[: len(freq) // 2], mtf[: len(mtf) // 2]
