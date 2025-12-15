"""Image segmentation algorithms for medical imaging."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import ndimage


@dataclass
class SegmentationResult:
    """Result of image segmentation."""

    labels: np.ndarray
    num_segments: int
    segment_properties: list[dict]
    mask: Optional[np.ndarray] = None


class Segmenter(ABC):
    """Abstract base class for segmentation algorithms."""

    @abstractmethod
    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Segment the image."""
        pass

    def _calculate_properties(
        self,
        labels: np.ndarray,
        image: np.ndarray,
    ) -> list[dict]:
        """Calculate properties for each segment."""
        properties = []
        unique_labels = np.unique(labels)

        for label in unique_labels:
            if label == 0:  # Skip background
                continue

            mask = labels == label
            coords = np.argwhere(mask)

            if len(coords) == 0:
                continue

            props = {
                "label": int(label),
                "area": int(mask.sum()),
                "centroid": tuple(coords.mean(axis=0)),
                "bounding_box": (
                    int(coords[:, 1].min()),
                    int(coords[:, 0].min()),
                    int(coords[:, 1].max() - coords[:, 1].min()),
                    int(coords[:, 0].max() - coords[:, 0].min()),
                ),
                "mean_intensity": float(image[mask].mean()),
                "std_intensity": float(image[mask].std()),
            }
            properties.append(props)

        return properties


class ThresholdSegmenter(Segmenter):
    """Threshold-based segmentation."""

    def __init__(
        self,
        method: str = "otsu",
        threshold: Optional[float] = None,
    ):
        self.method = method
        self.threshold = threshold

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Segment using thresholding."""
        if self.threshold is not None:
            thresh = self.threshold
        elif self.method == "otsu":
            thresh = self._otsu_threshold(image)
        elif self.method == "adaptive":
            return self._adaptive_threshold(image)
        else:
            thresh = image.mean()

        binary = image > thresh
        labels, num_features = ndimage.label(binary)

        return SegmentationResult(
            labels=labels,
            num_segments=num_features,
            segment_properties=self._calculate_properties(labels, image),
            mask=binary,
        )

    def _otsu_threshold(self, image: np.ndarray) -> float:
        """Calculate Otsu's threshold."""
        hist, bin_edges = np.histogram(image.flatten(), bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Class probabilities
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]

        # Class means
        mean1 = np.cumsum(hist * bin_centers) / (weight1 + 1e-10)
        mean2 = (np.cumsum((hist * bin_centers)[::-1]) / (weight2[::-1] + 1e-10))[::-1]

        # Between-class variance
        variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

        idx = np.argmax(variance)
        return bin_centers[idx]

    def _adaptive_threshold(self, image: np.ndarray) -> SegmentationResult:
        """Adaptive local thresholding."""
        block_size = 35
        offset = 10

        local_mean = ndimage.uniform_filter(image.astype(float), size=block_size)
        binary = image > (local_mean - offset)
        labels, num_features = ndimage.label(binary)

        return SegmentationResult(
            labels=labels,
            num_segments=num_features,
            segment_properties=self._calculate_properties(labels, image),
            mask=binary,
        )


class RegionGrowingSegmenter(Segmenter):
    """Region growing segmentation."""

    def __init__(
        self,
        seed_points: Optional[list[tuple[int, int]]] = None,
        threshold: float = 10.0,
        connectivity: int = 4,
    ):
        self.seed_points = seed_points
        self.threshold = threshold
        self.connectivity = connectivity

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Segment using region growing."""
        if self.seed_points is None:
            # Auto-detect seed points based on local maxima
            self.seed_points = self._find_seed_points(image)

        labels = np.zeros(image.shape, dtype=np.int32)
        current_label = 1

        for seed in self.seed_points:
            if labels[seed[0], seed[1]] == 0:
                self._grow_region(image, labels, seed, current_label)
                current_label += 1

        return SegmentationResult(
            labels=labels,
            num_segments=current_label - 1,
            segment_properties=self._calculate_properties(labels, image),
        )

    def _find_seed_points(self, image: np.ndarray, num_seeds: int = 5) -> list[tuple[int, int]]:
        """Find seed points automatically."""
        # Use local maxima as seeds
        local_max = ndimage.maximum_filter(image, size=20) == image
        coords = np.argwhere(local_max)

        if len(coords) > num_seeds:
            # Select brightest points
            intensities = image[local_max]
            indices = np.argsort(intensities)[-num_seeds:]
            coords = coords[indices]

        return [tuple(c) for c in coords]

    def _grow_region(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        seed: tuple[int, int],
        label: int,
    ):
        """Grow region from seed point."""
        h, w = image.shape
        seed_value = image[seed[0], seed[1]]

        # Define connectivity
        if self.connectivity == 4:
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                         (0, 1), (1, -1), (1, 0), (1, 1)]

        stack = [seed]
        labels[seed[0], seed[1]] = label

        while stack:
            y, x = stack.pop()

            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx

                if 0 <= ny < h and 0 <= nx < w:
                    if labels[ny, nx] == 0:
                        if abs(float(image[ny, nx]) - seed_value) <= self.threshold:
                            labels[ny, nx] = label
                            stack.append((ny, nx))


class WatershedSegmenter(Segmenter):
    """Watershed segmentation."""

    def __init__(
        self,
        markers: Optional[np.ndarray] = None,
        use_gradient: bool = True,
    ):
        self.markers = markers
        self.use_gradient = use_gradient

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Segment using watershed algorithm."""
        # Compute gradient if requested
        if self.use_gradient:
            gradient = self._compute_gradient(image)
        else:
            gradient = image

        # Generate markers if not provided
        if self.markers is None:
            markers = self._generate_markers(gradient)
        else:
            markers = self.markers

        # Simplified watershed
        labels = self._watershed(gradient, markers)

        return SegmentationResult(
            labels=labels,
            num_segments=labels.max(),
            segment_properties=self._calculate_properties(labels, image),
        )

    def _compute_gradient(self, image: np.ndarray) -> np.ndarray:
        """Compute morphological gradient."""
        dilated = ndimage.grey_dilation(image, size=(3, 3))
        eroded = ndimage.grey_erosion(image, size=(3, 3))
        return dilated - eroded

    def _generate_markers(self, gradient: np.ndarray) -> np.ndarray:
        """Generate markers for watershed."""
        # Find local minima
        local_min = ndimage.minimum_filter(gradient, size=20) == gradient

        # Label markers
        markers, _ = ndimage.label(local_min)
        return markers

    def _watershed(self, gradient: np.ndarray, markers: np.ndarray) -> np.ndarray:
        """Simplified watershed implementation."""
        # This is a simplified version - for production, use scipy.ndimage.watershed_ift
        labels = markers.copy()
        h, w = gradient.shape

        # Priority queue simulation
        changed = True
        while changed:
            changed = False
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if labels[y, x] == 0:
                        # Check neighbors
                        neighbor_labels = []
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dy == 0 and dx == 0:
                                    continue
                                nl = labels[y + dy, x + dx]
                                if nl > 0:
                                    neighbor_labels.append(nl)

                        if len(neighbor_labels) > 0:
                            # Assign most common neighbor label
                            labels[y, x] = max(set(neighbor_labels), key=neighbor_labels.count)
                            changed = True

        return labels


class MultiScaleSegmenter(Segmenter):
    """Multi-scale segmentation combining multiple approaches."""

    def __init__(self, scales: list[float] = None):
        self.scales = scales or [1.0, 0.5, 0.25]

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Perform multi-scale segmentation."""
        all_labels = []

        for scale in self.scales:
            # Resize image
            if scale != 1.0:
                scaled = ndimage.zoom(image, scale, order=1)
            else:
                scaled = image

            # Segment at this scale
            segmenter = ThresholdSegmenter(method="otsu")
            result = segmenter.segment(scaled)

            # Resize labels back
            if scale != 1.0:
                labels = ndimage.zoom(
                    result.labels.astype(float),
                    1.0 / scale,
                    order=0
                ).astype(np.int32)
            else:
                labels = result.labels

            all_labels.append(labels)

        # Combine results (use finest scale as base)
        final_labels = all_labels[0]

        return SegmentationResult(
            labels=final_labels,
            num_segments=final_labels.max(),
            segment_properties=self._calculate_properties(final_labels, image),
        )
