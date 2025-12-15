"""Image viewing and thumbnail generation."""

from dataclasses import dataclass, field
from typing import Optional, Callable, Any
import numpy as np


@dataclass
class ViewerSettings:
    """Image viewer settings."""

    window_center: float = 0.5
    window_width: float = 1.0
    zoom: float = 1.0
    pan_x: int = 0
    pan_y: int = 0
    rotation: int = 0  # degrees, 0/90/180/270
    flip_horizontal: bool = False
    flip_vertical: bool = False
    invert: bool = False
    colormap: str = "gray"


@dataclass
class Annotation:
    """Image annotation."""

    annotation_id: str
    annotation_type: str  # point, line, rectangle, ellipse, text, arrow
    coordinates: list = field(default_factory=list)
    label: str = ""
    color: str = "yellow"
    visible: bool = True
    measurements: dict = field(default_factory=dict)


class ImageViewer:
    """Medical image viewer."""

    def __init__(self):
        self.settings = ViewerSettings()
        self._image: Optional[np.ndarray] = None
        self._original: Optional[np.ndarray] = None
        self._annotations: list[Annotation] = []
        self._listeners: list[Callable[[str], None]] = []

    def add_listener(self, callback: Callable[[str], None]):
        """Add event listener."""
        self._listeners.append(callback)

    def _notify(self, event: str):
        """Notify listeners."""
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass

    def load_image(self, image: np.ndarray):
        """Load an image into the viewer."""
        self._original = image.copy()
        self._image = image.copy()
        self._apply_transforms()
        self._notify("image_loaded")

    def get_display_image(self) -> Optional[np.ndarray]:
        """Get the current display image."""
        return self._image

    def set_window(self, center: float, width: float):
        """Set window/level values."""
        self.settings.window_center = center
        self.settings.window_width = width
        self._apply_window()
        self._notify("window_changed")

    def set_zoom(self, zoom: float):
        """Set zoom level."""
        self.settings.zoom = max(0.1, min(10.0, zoom))
        self._notify("zoom_changed")

    def set_pan(self, x: int, y: int):
        """Set pan offset."""
        self.settings.pan_x = x
        self.settings.pan_y = y
        self._notify("pan_changed")

    def rotate(self, degrees: int):
        """Rotate image."""
        self.settings.rotation = (self.settings.rotation + degrees) % 360
        self._apply_transforms()
        self._notify("rotation_changed")

    def flip(self, horizontal: bool = False, vertical: bool = False):
        """Flip image."""
        if horizontal:
            self.settings.flip_horizontal = not self.settings.flip_horizontal
        if vertical:
            self.settings.flip_vertical = not self.settings.flip_vertical
        self._apply_transforms()
        self._notify("flip_changed")

    def invert(self):
        """Invert image."""
        self.settings.invert = not self.settings.invert
        self._apply_window()
        self._notify("invert_changed")

    def reset(self):
        """Reset all transformations."""
        self.settings = ViewerSettings()
        if self._original is not None:
            self._image = self._original.copy()
        self._notify("reset")

    def _apply_transforms(self):
        """Apply geometric transformations."""
        if self._original is None:
            return

        img = self._original.copy()

        # Rotation
        if self.settings.rotation == 90:
            img = np.rot90(img, 1)
        elif self.settings.rotation == 180:
            img = np.rot90(img, 2)
        elif self.settings.rotation == 270:
            img = np.rot90(img, 3)

        # Flip
        if self.settings.flip_horizontal:
            img = np.fliplr(img)
        if self.settings.flip_vertical:
            img = np.flipud(img)

        self._image = img
        self._apply_window()

    def _apply_window(self):
        """Apply window/level transformation."""
        if self._image is None:
            return

        img = self._image.astype(float)

        # Normalize to 0-1
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)

        # Apply window
        wc = self.settings.window_center
        ww = self.settings.window_width
        img = (img - (wc - ww / 2)) / ww
        img = np.clip(img, 0, 1)

        # Invert if needed
        if self.settings.invert:
            img = 1 - img

        self._image = (img * 255).astype(np.uint8)

    # Annotations
    def add_annotation(self, annotation: Annotation):
        """Add an annotation."""
        self._annotations.append(annotation)
        self._notify("annotation_added")

    def remove_annotation(self, annotation_id: str):
        """Remove an annotation."""
        self._annotations = [a for a in self._annotations if a.annotation_id != annotation_id]
        self._notify("annotation_removed")

    def get_annotations(self) -> list[Annotation]:
        """Get all annotations."""
        return self._annotations

    def clear_annotations(self):
        """Clear all annotations."""
        self._annotations.clear()
        self._notify("annotations_cleared")

    # Measurements
    def measure_distance(
        self,
        point1: tuple[int, int],
        point2: tuple[int, int],
        pixel_spacing: float = 1.0,
    ) -> float:
        """Measure distance between two points."""
        dx = (point2[0] - point1[0]) * pixel_spacing
        dy = (point2[1] - point1[1]) * pixel_spacing
        return np.sqrt(dx * dx + dy * dy)

    def measure_roi_stats(
        self,
        roi_mask: np.ndarray,
    ) -> dict:
        """Calculate statistics in a region of interest."""
        if self._image is None:
            return {}

        roi_values = self._image[roi_mask]
        return {
            "mean": float(roi_values.mean()),
            "std": float(roi_values.std()),
            "min": float(roi_values.min()),
            "max": float(roi_values.max()),
            "area_pixels": int(roi_values.size),
        }

    def get_pixel_value(self, x: int, y: int) -> Optional[float]:
        """Get pixel value at coordinates."""
        if self._image is None:
            return None
        if 0 <= y < self._image.shape[0] and 0 <= x < self._image.shape[1]:
            return float(self._image[y, x])
        return None


class ThumbnailGenerator:
    """Generate thumbnails for images."""

    def __init__(self, default_size: tuple[int, int] = (128, 128)):
        self.default_size = default_size

    def generate(
        self,
        image: np.ndarray,
        size: tuple[int, int] = None,
        maintain_aspect: bool = True,
    ) -> np.ndarray:
        """Generate a thumbnail from an image."""
        target_size = size or self.default_size

        h, w = image.shape[:2]
        target_h, target_w = target_size

        if maintain_aspect:
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
        else:
            new_w, new_h = target_w, target_h

        # Simple resize using slicing (basic downsampling)
        # For better quality, you'd use scipy.ndimage.zoom or PIL
        step_y = max(1, h // new_h)
        step_x = max(1, w // new_w)
        thumbnail = image[::step_y, ::step_x]

        # Ensure exact size
        thumbnail = thumbnail[:new_h, :new_w]

        return thumbnail

    def generate_strip(
        self,
        images: list[np.ndarray],
        thumbnail_size: tuple[int, int] = None,
        max_images: int = 10,
    ) -> np.ndarray:
        """Generate a horizontal strip of thumbnails."""
        size = thumbnail_size or self.default_size
        images = images[:max_images]

        thumbnails = [self.generate(img, size) for img in images]

        # Create strip
        strip_width = size[0] * len(thumbnails)
        strip_height = size[1]
        strip = np.zeros((strip_height, strip_width), dtype=np.uint8)

        for i, thumb in enumerate(thumbnails):
            x_offset = i * size[0]
            h, w = thumb.shape[:2]
            strip[:h, x_offset:x_offset + w] = thumb

        return strip

    def generate_grid(
        self,
        images: list[np.ndarray],
        grid_size: tuple[int, int] = (3, 3),
        thumbnail_size: tuple[int, int] = None,
    ) -> np.ndarray:
        """Generate a grid of thumbnails."""
        size = thumbnail_size or self.default_size
        rows, cols = grid_size
        max_images = rows * cols
        images = images[:max_images]

        thumbnails = [self.generate(img, size) for img in images]

        # Create grid
        grid_width = size[0] * cols
        grid_height = size[1] * rows
        grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

        for i, thumb in enumerate(thumbnails):
            row = i // cols
            col = i % cols
            x_offset = col * size[0]
            y_offset = row * size[1]
            h, w = thumb.shape[:2]
            grid[y_offset:y_offset + h, x_offset:x_offset + w] = thumb

        return grid


class SeriesViewer:
    """View a series of images."""

    def __init__(self):
        self.viewer = ImageViewer()
        self._images: list[np.ndarray] = []
        self._current_index: int = 0
        self._frame_rate: float = 10.0  # fps for cine
        self._is_playing: bool = False
        self._listeners: list[Callable[[int, int], None]] = []

    def add_listener(self, callback: Callable[[int, int], None]):
        """Add frame change listener."""
        self._listeners.append(callback)

    def _notify(self, index: int, total: int):
        """Notify listeners."""
        for listener in self._listeners:
            try:
                listener(index, total)
            except Exception:
                pass

    def load_series(self, images: list[np.ndarray]):
        """Load a series of images."""
        self._images = images
        self._current_index = 0
        if images:
            self.viewer.load_image(images[0])
        self._notify(0, len(images))

    def get_current_index(self) -> int:
        """Get current image index."""
        return self._current_index

    def get_total_images(self) -> int:
        """Get total number of images."""
        return len(self._images)

    def go_to(self, index: int):
        """Go to a specific image."""
        if 0 <= index < len(self._images):
            self._current_index = index
            self.viewer.load_image(self._images[index])
            self._notify(index, len(self._images))

    def next(self):
        """Go to next image."""
        if self._current_index < len(self._images) - 1:
            self.go_to(self._current_index + 1)

    def previous(self):
        """Go to previous image."""
        if self._current_index > 0:
            self.go_to(self._current_index - 1)

    def first(self):
        """Go to first image."""
        self.go_to(0)

    def last(self):
        """Go to last image."""
        self.go_to(len(self._images) - 1)

    def set_frame_rate(self, fps: float):
        """Set cine frame rate."""
        self._frame_rate = max(1.0, min(60.0, fps))

    def play(self):
        """Start cine playback."""
        self._is_playing = True

    def stop(self):
        """Stop cine playback."""
        self._is_playing = False

    def is_playing(self) -> bool:
        """Check if cine is playing."""
        return self._is_playing
