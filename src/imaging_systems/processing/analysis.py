"""Image analysis tools for medical imaging."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any
import numpy as np
from scipy import ndimage


class ROIShape(Enum):
    """Region of Interest shapes."""

    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"
    POLYGON = "polygon"
    FREEHAND = "freehand"


@dataclass
class RegionOfInterest:
    """Region of Interest definition."""

    roi_id: str
    shape: ROIShape
    coordinates: list  # [(x, y), ...] for polygon, (x, y, w, h) for rect
    label: str = ""
    color: str = "#FF0000"
    metadata: dict = field(default_factory=dict)

    def get_mask(self, image_shape: tuple[int, int]) -> np.ndarray:
        """Generate binary mask for the ROI."""
        mask = np.zeros(image_shape, dtype=bool)
        h, w = image_shape

        if self.shape == ROIShape.RECTANGLE:
            x, y, roi_w, roi_h = self.coordinates
            x, y = int(x), int(y)
            roi_w, roi_h = int(roi_w), int(roi_h)
            mask[y:y + roi_h, x:x + roi_w] = True

        elif self.shape == ROIShape.ELLIPSE:
            x, y, rx, ry = self.coordinates
            yy, xx = np.ogrid[:h, :w]
            ellipse = ((xx - x) / rx) ** 2 + ((yy - y) / ry) ** 2 <= 1
            mask = ellipse

        elif self.shape in (ROIShape.POLYGON, ROIShape.FREEHAND):
            # Simple polygon fill
            from matplotlib.path import Path
            coords = np.array(self.coordinates)
            path = Path(coords)
            yy, xx = np.mgrid[:h, :w]
            points = np.column_stack((xx.flatten(), yy.flatten()))
            mask = path.contains_points(points).reshape(image_shape)

        return mask

    def get_area(self, pixel_spacing: tuple[float, float] = (1.0, 1.0)) -> float:
        """Calculate area in physical units."""
        if self.shape == ROIShape.RECTANGLE:
            _, _, w, h = self.coordinates
            return w * h * pixel_spacing[0] * pixel_spacing[1]
        elif self.shape == ROIShape.ELLIPSE:
            _, _, rx, ry = self.coordinates
            return np.pi * rx * ry * pixel_spacing[0] * pixel_spacing[1]
        else:
            # For polygon, calculate from mask
            return len(self.coordinates) * pixel_spacing[0] * pixel_spacing[1]


@dataclass
class MeasurementResult:
    """Result of a measurement."""

    measurement_type: str
    value: float
    unit: str
    location: Optional[tuple] = None
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class MeasurementTool:
    """Tools for making measurements on medical images."""

    def __init__(self, pixel_spacing: tuple[float, float] = (1.0, 1.0)):
        """Initialize with pixel spacing in mm."""
        self.pixel_spacing = pixel_spacing

    def measure_distance(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float],
    ) -> MeasurementResult:
        """Measure distance between two points."""
        dx = (point2[0] - point1[0]) * self.pixel_spacing[0]
        dy = (point2[1] - point1[1]) * self.pixel_spacing[1]
        distance = np.sqrt(dx ** 2 + dy ** 2)

        return MeasurementResult(
            measurement_type="distance",
            value=distance,
            unit="mm",
            location=(point1, point2),
        )

    def measure_angle(
        self,
        vertex: tuple[float, float],
        point1: tuple[float, float],
        point2: tuple[float, float],
    ) -> MeasurementResult:
        """Measure angle between three points."""
        v1 = np.array([point1[0] - vertex[0], point1[1] - vertex[1]])
        v2 = np.array([point2[0] - vertex[0], point2[1] - vertex[1]])

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

        return MeasurementResult(
            measurement_type="angle",
            value=angle,
            unit="degrees",
            location=(vertex, point1, point2),
        )

    def measure_area(self, roi: RegionOfInterest) -> MeasurementResult:
        """Measure area of a region."""
        area = roi.get_area(self.pixel_spacing)

        return MeasurementResult(
            measurement_type="area",
            value=area,
            unit="mm²",
            metadata={"roi_id": roi.roi_id},
        )

    def measure_perimeter(
        self,
        points: list[tuple[float, float]],
        closed: bool = True,
    ) -> MeasurementResult:
        """Measure perimeter of a polygon."""
        perimeter = 0.0
        n = len(points)

        for i in range(n):
            next_i = (i + 1) % n if closed else min(i + 1, n - 1)
            if next_i != i:
                dx = (points[next_i][0] - points[i][0]) * self.pixel_spacing[0]
                dy = (points[next_i][1] - points[i][1]) * self.pixel_spacing[1]
                perimeter += np.sqrt(dx ** 2 + dy ** 2)

        return MeasurementResult(
            measurement_type="perimeter",
            value=perimeter,
            unit="mm",
        )

    def measure_roi_statistics(
        self,
        image: np.ndarray,
        roi: RegionOfInterest,
    ) -> dict[str, MeasurementResult]:
        """Calculate statistics within a region."""
        mask = roi.get_mask(image.shape[:2])
        values = image[mask]

        if len(values) == 0:
            return {}

        results = {
            "mean": MeasurementResult("mean", float(values.mean()), "intensity"),
            "std": MeasurementResult("std", float(values.std()), "intensity"),
            "min": MeasurementResult("min", float(values.min()), "intensity"),
            "max": MeasurementResult("max", float(values.max()), "intensity"),
            "median": MeasurementResult("median", float(np.median(values)), "intensity"),
            "area": self.measure_area(roi),
        }

        return results


@dataclass
class HistogramAnalysis:
    """Histogram analysis results."""

    histogram: np.ndarray
    bin_edges: np.ndarray
    mean: float
    std: float
    mode: float
    percentiles: dict[int, float]

    @classmethod
    def compute(
        cls,
        image: np.ndarray,
        bins: int = 256,
        mask: Optional[np.ndarray] = None,
    ) -> "HistogramAnalysis":
        """Compute histogram analysis."""
        if mask is not None:
            data = image[mask]
        else:
            data = image.flatten()

        hist, edges = np.histogram(data, bins=bins)

        # Find mode (most frequent value)
        mode_idx = np.argmax(hist)
        mode = (edges[mode_idx] + edges[mode_idx + 1]) / 2

        # Calculate percentiles
        percentiles = {
            1: float(np.percentile(data, 1)),
            5: float(np.percentile(data, 5)),
            25: float(np.percentile(data, 25)),
            50: float(np.percentile(data, 50)),
            75: float(np.percentile(data, 75)),
            95: float(np.percentile(data, 95)),
            99: float(np.percentile(data, 99)),
        }

        return cls(
            histogram=hist,
            bin_edges=edges,
            mean=float(data.mean()),
            std=float(data.std()),
            mode=mode,
            percentiles=percentiles,
        )


class ImageAnalyzer:
    """Comprehensive image analyzer."""

    def __init__(
        self,
        pixel_spacing: tuple[float, float] = (1.0, 1.0),
    ):
        self.pixel_spacing = pixel_spacing
        self.measurement_tool = MeasurementTool(pixel_spacing)
        self._rois: list[RegionOfInterest] = []

    def add_roi(self, roi: RegionOfInterest):
        """Add a region of interest."""
        self._rois.append(roi)

    def remove_roi(self, roi_id: str):
        """Remove a region of interest."""
        self._rois = [r for r in self._rois if r.roi_id != roi_id]

    def analyze_image(self, image: np.ndarray) -> dict[str, Any]:
        """Perform comprehensive image analysis."""
        results = {
            "image_shape": image.shape,
            "dtype": str(image.dtype),
            "global_statistics": self._compute_global_statistics(image),
            "histogram": HistogramAnalysis.compute(image),
            "roi_analyses": {},
        }

        # Analyze each ROI
        for roi in self._rois:
            results["roi_analyses"][roi.roi_id] = self.measurement_tool.measure_roi_statistics(image, roi)

        return results

    def _compute_global_statistics(self, image: np.ndarray) -> dict:
        """Compute global image statistics."""
        return {
            "mean": float(image.mean()),
            "std": float(image.std()),
            "min": float(image.min()),
            "max": float(image.max()),
            "median": float(np.median(image)),
            "snr": float(image.mean() / (image.std() + 1e-10)),
        }

    def detect_edges(self, image: np.ndarray, method: str = "canny") -> np.ndarray:
        """Detect edges in image."""
        if method == "canny":
            # Simplified Canny-like edge detection
            sobel_x = ndimage.sobel(image.astype(float), axis=0)
            sobel_y = ndimage.sobel(image.astype(float), axis=1)
            magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

            # Non-maximum suppression (simplified)
            threshold = magnitude.mean() + magnitude.std()
            edges = magnitude > threshold
            return edges.astype(np.uint8) * 255

        elif method == "sobel":
            sobel_x = ndimage.sobel(image.astype(float), axis=0)
            sobel_y = ndimage.sobel(image.astype(float), axis=1)
            return np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        elif method == "laplacian":
            return np.abs(ndimage.laplace(image.astype(float)))

        return image

    def find_contours(
        self,
        binary_image: np.ndarray,
    ) -> list[np.ndarray]:
        """Find contours in binary image."""
        # Label connected regions
        labeled, num_features = ndimage.label(binary_image)

        contours = []
        for i in range(1, num_features + 1):
            region = labeled == i
            # Find boundary pixels
            eroded = ndimage.binary_erosion(region)
            boundary = region & ~eroded
            coords = np.argwhere(boundary)
            if len(coords) > 0:
                contours.append(coords)

        return contours

    def calculate_texture_features(
        self,
        image: np.ndarray,
        roi: Optional[RegionOfInterest] = None,
    ) -> dict[str, float]:
        """Calculate texture features."""
        if roi is not None:
            mask = roi.get_mask(image.shape[:2])
            data = image[mask]
        else:
            data = image.flatten()

        # Basic texture features
        features = {
            "contrast": float(data.std() ** 2),
            "energy": float((data ** 2).sum() / len(data)),
            "entropy": self._calculate_entropy(data),
            "homogeneity": self._calculate_homogeneity(data),
        }

        return features

    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy."""
        hist, _ = np.histogram(data, bins=256, density=True)
        hist = hist[hist > 0]  # Remove zeros
        return -float(np.sum(hist * np.log2(hist + 1e-10)))

    def _calculate_homogeneity(self, data: np.ndarray) -> float:
        """Calculate homogeneity (inverse difference moment)."""
        # Simplified calculation
        diff = np.abs(np.diff(data))
        return float(1 / (1 + diff.mean()))
