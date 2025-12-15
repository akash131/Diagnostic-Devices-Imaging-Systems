"""Image filters for medical imaging processing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import ndimage
from scipy.signal import wiener


@dataclass
class FilterParameters:
    """Common filter parameters."""

    kernel_size: int = 3
    sigma: float = 1.0
    strength: float = 1.0
    iterations: int = 1


class ImageFilter(ABC):
    """Abstract base class for image filters."""

    def __init__(self, params: Optional[FilterParameters] = None):
        self.params = params or FilterParameters()

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the filter to an image."""
        pass

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Allow filter to be called directly."""
        return self.apply(image)


class NoiseReductionFilter(ImageFilter):
    """Noise reduction filters for medical images."""

    def __init__(
        self,
        method: str = "gaussian",
        params: Optional[FilterParameters] = None,
    ):
        super().__init__(params)
        self.method = method

    def apply(self, image: np.ndarray) -> np.ndarray:
        if self.method == "gaussian":
            return self._gaussian_filter(image)
        elif self.method == "median":
            return self._median_filter(image)
        elif self.method == "bilateral":
            return self._bilateral_filter(image)
        elif self.method == "wiener":
            return self._wiener_filter(image)
        elif self.method == "nlm":
            return self._non_local_means(image)
        else:
            return image

    def _gaussian_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian smoothing."""
        return ndimage.gaussian_filter(image, sigma=self.params.sigma)

    def _median_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply median filter."""
        return ndimage.median_filter(image, size=self.params.kernel_size)

    def _bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply bilateral filter (edge-preserving smoothing)."""
        # Simplified bilateral filter implementation
        sigma_space = self.params.sigma
        sigma_color = self.params.sigma * 30

        filtered = ndimage.gaussian_filter(image.astype(float), sigma_space)
        edges = np.abs(image.astype(float) - filtered)
        weight = np.exp(-edges ** 2 / (2 * sigma_color ** 2))

        return (image * (1 - self.params.strength * weight) +
                filtered * self.params.strength * weight)

    def _wiener_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply Wiener filter."""
        return wiener(image.astype(float), mysize=self.params.kernel_size)

    def _non_local_means(self, image: np.ndarray) -> np.ndarray:
        """Simplified non-local means denoising."""
        # Use Gaussian as fallback for simplified implementation
        return ndimage.gaussian_filter(image, sigma=self.params.sigma * 0.5)


class EdgeEnhancementFilter(ImageFilter):
    """Edge enhancement filters."""

    def __init__(
        self,
        method: str = "unsharp_mask",
        params: Optional[FilterParameters] = None,
    ):
        super().__init__(params)
        self.method = method

    def apply(self, image: np.ndarray) -> np.ndarray:
        if self.method == "unsharp_mask":
            return self._unsharp_mask(image)
        elif self.method == "laplacian":
            return self._laplacian_enhancement(image)
        elif self.method == "sobel":
            return self._sobel_enhancement(image)
        else:
            return image

    def _unsharp_mask(self, image: np.ndarray) -> np.ndarray:
        """Apply unsharp masking."""
        blurred = ndimage.gaussian_filter(image.astype(float), sigma=self.params.sigma)
        mask = image.astype(float) - blurred
        return np.clip(image + self.params.strength * mask, 0, image.max())

    def _laplacian_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply Laplacian edge enhancement."""
        laplacian = ndimage.laplace(image.astype(float))
        return np.clip(image - self.params.strength * laplacian, 0, image.max())

    def _sobel_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply Sobel edge enhancement."""
        sobel_x = ndimage.sobel(image.astype(float), axis=0)
        sobel_y = ndimage.sobel(image.astype(float), axis=1)
        edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        return np.clip(image + self.params.strength * edges * 0.1, 0, image.max())


class ContrastEnhancementFilter(ImageFilter):
    """Contrast enhancement filters."""

    def __init__(
        self,
        method: str = "clahe",
        params: Optional[FilterParameters] = None,
    ):
        super().__init__(params)
        self.method = method

    def apply(self, image: np.ndarray) -> np.ndarray:
        if self.method == "clahe":
            return self._clahe(image)
        elif self.method == "histogram_equalization":
            return self._histogram_equalization(image)
        elif self.method == "gamma":
            return self._gamma_correction(image)
        elif self.method == "sigmoid":
            return self._sigmoid_correction(image)
        else:
            return image

    def _clahe(self, image: np.ndarray) -> np.ndarray:
        """Contrast Limited Adaptive Histogram Equalization."""
        # Simplified CLAHE implementation
        clip_limit = 2.0 * self.params.strength
        tile_size = self.params.kernel_size * 8

        # Normalize to 0-255
        normalized = ((image - image.min()) / (image.max() - image.min() + 1e-10) * 255)

        # Apply local histogram equalization
        result = np.zeros_like(normalized)
        h, w = normalized.shape[:2]

        for i in range(0, h, tile_size):
            for j in range(0, w, tile_size):
                tile = normalized[i:i + tile_size, j:j + tile_size]
                if tile.size > 0:
                    hist, bins = np.histogram(tile.flatten(), bins=256, range=(0, 256))
                    hist = np.clip(hist, 0, clip_limit * tile.size / 256)
                    cdf = hist.cumsum()
                    cdf = cdf / cdf[-1] * 255
                    result[i:i + tile_size, j:j + tile_size] = cdf[tile.astype(int)]

        return result.astype(image.dtype)

    def _histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Global histogram equalization."""
        flat = image.flatten()
        hist, bins = np.histogram(flat, bins=256, range=(flat.min(), flat.max()))
        cdf = hist.cumsum()
        cdf = (cdf - cdf.min()) / (cdf[-1] - cdf.min()) * (flat.max() - flat.min()) + flat.min()

        return cdf[((flat - flat.min()) / (flat.max() - flat.min() + 1e-10) * 255).astype(int)].reshape(image.shape)

    def _gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply gamma correction."""
        gamma = 1.0 / (1.0 + self.params.strength * 0.5)
        normalized = (image - image.min()) / (image.max() - image.min() + 1e-10)
        corrected = np.power(normalized, gamma)
        return corrected * (image.max() - image.min()) + image.min()

    def _sigmoid_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply sigmoid contrast correction."""
        normalized = (image - image.min()) / (image.max() - image.min() + 1e-10)
        gain = 10 * self.params.strength
        cutoff = 0.5
        corrected = 1 / (1 + np.exp(gain * (cutoff - normalized)))
        return corrected * (image.max() - image.min()) + image.min()


class SharpeningFilter(ImageFilter):
    """Image sharpening filters."""

    def __init__(self, params: Optional[FilterParameters] = None):
        super().__init__(params)

    def apply(self, image: np.ndarray) -> np.ndarray:
        # Unsharp masking-based sharpening
        blurred = ndimage.gaussian_filter(image.astype(float), sigma=self.params.sigma)
        sharpened = image + self.params.strength * (image - blurred)
        return np.clip(sharpened, image.min(), image.max()).astype(image.dtype)


class SmoothingFilter(ImageFilter):
    """Image smoothing filters."""

    def __init__(
        self,
        method: str = "gaussian",
        params: Optional[FilterParameters] = None,
    ):
        super().__init__(params)
        self.method = method

    def apply(self, image: np.ndarray) -> np.ndarray:
        if self.method == "gaussian":
            return ndimage.gaussian_filter(image, sigma=self.params.sigma)
        elif self.method == "uniform":
            return ndimage.uniform_filter(image, size=self.params.kernel_size)
        elif self.method == "median":
            return ndimage.median_filter(image, size=self.params.kernel_size)
        else:
            return image


class ModalitySpecificFilter(ImageFilter):
    """Modality-specific filters."""

    def __init__(
        self,
        modality: str,
        params: Optional[FilterParameters] = None,
    ):
        super().__init__(params)
        self.modality = modality

    def apply(self, image: np.ndarray) -> np.ndarray:
        if self.modality == "ultrasound":
            return self._process_ultrasound(image)
        elif self.modality == "xray":
            return self._process_xray(image)
        elif self.modality == "oct":
            return self._process_oct(image)
        elif self.modality == "thermal":
            return self._process_thermal(image)
        else:
            return image

    def _process_ultrasound(self, image: np.ndarray) -> np.ndarray:
        """Speckle reduction for ultrasound."""
        # Lee filter for speckle reduction
        mean = ndimage.uniform_filter(image.astype(float), size=self.params.kernel_size)
        var = ndimage.uniform_filter(image.astype(float) ** 2, size=self.params.kernel_size) - mean ** 2
        noise_var = var.mean()
        weight = var / (var + noise_var + 1e-10)
        return mean + weight * (image - mean)

    def _process_xray(self, image: np.ndarray) -> np.ndarray:
        """Edge-preserving smoothing for X-ray."""
        bilateral = NoiseReductionFilter(method="bilateral", params=self.params)
        return bilateral.apply(image)

    def _process_oct(self, image: np.ndarray) -> np.ndarray:
        """Speckle reduction for OCT."""
        median = ndimage.median_filter(image, size=self.params.kernel_size)
        return median

    def _process_thermal(self, image: np.ndarray) -> np.ndarray:
        """Smoothing for thermal images."""
        return ndimage.gaussian_filter(image, sigma=self.params.sigma)
