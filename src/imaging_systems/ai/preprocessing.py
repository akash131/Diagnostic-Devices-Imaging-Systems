"""Image preprocessing for AI models."""

from dataclasses import dataclass, field
from typing import Optional, Callable, Any
import numpy as np
from scipy import ndimage


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""

    target_size: tuple[int, int] = (224, 224)
    normalize: bool = True
    normalize_range: tuple[float, float] = (0.0, 1.0)
    mean: Optional[float] = None
    std: Optional[float] = None
    channel_order: str = "channels_last"  # channels_last or channels_first
    grayscale: bool = True
    dtype: str = "float32"


class Preprocessor:
    """Preprocess images for AI model input."""

    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline."""
        img = image.copy()

        # Convert to grayscale if needed
        if self.config.grayscale and len(img.shape) > 2 and img.shape[-1] > 1:
            img = img.mean(axis=-1)

        # Resize
        img = self._resize(img, self.config.target_size)

        # Normalize
        if self.config.normalize:
            img = self._normalize(img)

        # Apply mean/std normalization
        if self.config.mean is not None:
            img = img - self.config.mean
        if self.config.std is not None:
            img = img / self.config.std

        # Add channel dimension
        if self.config.grayscale:
            if self.config.channel_order == "channels_last":
                img = img[..., np.newaxis]
            else:
                img = img[np.newaxis, ...]

        # Convert dtype
        img = img.astype(self.config.dtype)

        return img

    def _resize(self, image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """Resize image to target size."""
        h, w = image.shape[:2]
        target_h, target_w = target_size

        if h == target_h and w == target_w:
            return image

        # Calculate zoom factors
        zoom_h = target_h / h
        zoom_w = target_w / w

        if len(image.shape) == 2:
            return ndimage.zoom(image, (zoom_h, zoom_w), order=1)
        else:
            return ndimage.zoom(image, (zoom_h, zoom_w, 1), order=1)

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to range."""
        img = image.astype(np.float32)
        min_val, max_val = self.config.normalize_range

        img_min = img.min()
        img_max = img.max()

        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
            img = img * (max_val - min_val) + min_val

        return img

    def batch_process(self, images: list[np.ndarray]) -> np.ndarray:
        """Process a batch of images."""
        processed = [self.process(img) for img in images]
        return np.stack(processed)


@dataclass
class AugmentationConfig:
    """Augmentation configuration."""

    enabled: bool = True
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation_range: float = 15.0  # degrees
    brightness_range: tuple[float, float] = (0.9, 1.1)
    contrast_range: tuple[float, float] = (0.9, 1.1)
    noise_std: float = 0.01
    zoom_range: tuple[float, float] = (0.95, 1.05)


class AugmentationPipeline:
    """Image augmentation for training."""

    def __init__(self, config: AugmentationConfig = None):
        self.config = config or AugmentationConfig()
        self._rng = np.random.default_rng()

    def augment(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation pipeline."""
        if not self.config.enabled:
            return image

        img = image.copy()

        # Horizontal flip
        if self.config.horizontal_flip and self._rng.random() > 0.5:
            img = np.fliplr(img)

        # Vertical flip
        if self.config.vertical_flip and self._rng.random() > 0.5:
            img = np.flipud(img)

        # Rotation
        if self.config.rotation_range > 0:
            angle = self._rng.uniform(-self.config.rotation_range, self.config.rotation_range)
            img = ndimage.rotate(img, angle, reshape=False, mode="nearest")

        # Brightness
        if self.config.brightness_range != (1.0, 1.0):
            factor = self._rng.uniform(*self.config.brightness_range)
            img = img * factor
            img = np.clip(img, 0, img.max())

        # Contrast
        if self.config.contrast_range != (1.0, 1.0):
            factor = self._rng.uniform(*self.config.contrast_range)
            mean = img.mean()
            img = (img - mean) * factor + mean
            img = np.clip(img, 0, img.max())

        # Noise
        if self.config.noise_std > 0:
            noise = self._rng.normal(0, self.config.noise_std, img.shape)
            img = img + noise * img.max()
            img = np.clip(img, 0, img.max())

        return img

    def batch_augment(
        self,
        images: list[np.ndarray],
        augmentations_per_image: int = 1,
    ) -> list[np.ndarray]:
        """Augment a batch of images."""
        augmented = []
        for img in images:
            augmented.append(img)  # Keep original
            for _ in range(augmentations_per_image):
                augmented.append(self.augment(img))
        return augmented


class MedicalImagePreprocessor(Preprocessor):
    """Specialized preprocessor for medical images."""

    def __init__(
        self,
        modality: str = "",
        config: PreprocessingConfig = None,
    ):
        super().__init__(config)
        self.modality = modality

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply medical image-specific preprocessing."""
        img = image.copy()

        # Apply modality-specific preprocessing
        if self.modality == "US":
            img = self._preprocess_ultrasound(img)
        elif self.modality in ("DX", "CR"):
            img = self._preprocess_xray(img)
        elif self.modality == "OPT":
            img = self._preprocess_oct(img)
        elif self.modality == "TG":
            img = self._preprocess_thermal(img)

        # Apply standard preprocessing
        return super().process(img)

    def _preprocess_ultrasound(self, image: np.ndarray) -> np.ndarray:
        """Preprocess ultrasound image."""
        # Remove speckle noise using median filter
        img = ndimage.median_filter(image, size=3)

        # Enhance contrast
        p2, p98 = np.percentile(img, (2, 98))
        img = np.clip(img, p2, p98)

        return img

    def _preprocess_xray(self, image: np.ndarray) -> np.ndarray:
        """Preprocess X-ray image."""
        # Window/level adjustment
        img = image.astype(np.float32)

        # Apply CLAHE-like enhancement
        # Normalize first
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)

        # Apply histogram equalization
        hist, bins = np.histogram(img.flatten(), bins=256, range=(0, 1))
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]

        # Map values
        img_flat = img.flatten()
        img_eq = np.interp(img_flat, bins[:-1], cdf_normalized)
        img = img_eq.reshape(img.shape)

        return img

    def _preprocess_oct(self, image: np.ndarray) -> np.ndarray:
        """Preprocess OCT image."""
        # Reduce speckle
        img = ndimage.uniform_filter(image, size=3)

        # Log compression (common for OCT)
        img = np.log1p(img.astype(np.float32))

        return img

    def _preprocess_thermal(self, image: np.ndarray) -> np.ndarray:
        """Preprocess thermal image."""
        # Thermal images often need temperature normalization
        img = image.astype(np.float32)

        # Normalize to typical body temperature range
        # Assuming input is in Celsius
        temp_min = 30.0  # Expected min temp
        temp_max = 40.0  # Expected max temp

        img = np.clip(img, temp_min, temp_max)
        img = (img - temp_min) / (temp_max - temp_min)

        return img


class InputValidator:
    """Validate images before AI inference."""

    def __init__(
        self,
        min_size: tuple[int, int] = (32, 32),
        max_size: tuple[int, int] = (4096, 4096),
        min_value: float = 0,
        max_value: float = 65535,
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, image: np.ndarray) -> tuple[bool, list[str]]:
        """Validate image for AI inference."""
        issues = []

        # Check dimensions
        if len(image.shape) < 2:
            issues.append("Image must be at least 2D")
            return False, issues

        h, w = image.shape[:2]

        # Check size
        if h < self.min_size[0] or w < self.min_size[1]:
            issues.append(f"Image too small: {w}x{h}, minimum: {self.min_size[1]}x{self.min_size[0]}")

        if h > self.max_size[0] or w > self.max_size[1]:
            issues.append(f"Image too large: {w}x{h}, maximum: {self.max_size[1]}x{self.max_size[0]}")

        # Check values
        if image.min() < self.min_value:
            issues.append(f"Image contains values below {self.min_value}")

        if image.max() > self.max_value:
            issues.append(f"Image contains values above {self.max_value}")

        # Check for NaN or Inf
        if np.isnan(image).any():
            issues.append("Image contains NaN values")

        if np.isinf(image).any():
            issues.append("Image contains infinite values")

        # Check for empty/blank image
        if image.std() < 1e-6:
            issues.append("Image appears to be blank or constant")

        return len(issues) == 0, issues

    def quality_score(self, image: np.ndarray) -> float:
        """Calculate image quality score (0-1)."""
        score = 1.0

        # Check contrast
        contrast = image.std() / (image.mean() + 1e-10)
        if contrast < 0.1:
            score -= 0.3
        elif contrast < 0.2:
            score -= 0.1

        # Check for saturation
        if image.dtype == np.uint8:
            saturated = np.sum((image == 0) | (image == 255)) / image.size
        else:
            saturated = np.sum((image == image.min()) | (image == image.max())) / image.size

        if saturated > 0.1:
            score -= 0.2

        # Check for noise (simple estimation)
        laplacian = ndimage.laplace(image.astype(np.float32))
        noise_estimate = laplacian.std()
        if noise_estimate > image.std() * 0.5:
            score -= 0.2

        return max(0.0, min(1.0, score))
