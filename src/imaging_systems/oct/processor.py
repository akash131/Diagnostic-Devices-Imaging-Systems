"""OCT signal processing implementation."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d


@dataclass
class ProcessingConfig:
    """OCT processing configuration."""

    zero_padding_factor: int = 2
    apply_dispersion_compensation: bool = True
    apply_spectral_shaping: bool = True
    window_type: str = "hanning"
    background_subtraction: bool = True


class SpectralProcessor:
    """Spectral domain OCT signal processor."""

    def __init__(
        self,
        num_pixels: int = 2048,
        wavelength_range: tuple[float, float] = (800, 880),
    ):
        self.num_pixels = num_pixels
        self.wavelength_range = wavelength_range

        self._config = ProcessingConfig()
        self._wavelength_to_k_map: Optional[np.ndarray] = None
        self._dispersion_coeffs: Optional[np.ndarray] = None
        self._background_spectrum: Optional[np.ndarray] = None
        self._window: Optional[np.ndarray] = None

        self._initialize_processing()

    def _initialize_processing(self):
        """Initialize processing parameters."""
        # Generate wavelength to wavenumber mapping
        wavelengths = np.linspace(
            self.wavelength_range[0],
            self.wavelength_range[1],
            self.num_pixels,
        )

        # Convert to wavenumber (k = 2π/λ)
        k = 2 * np.pi / wavelengths

        # Generate uniform k-space sampling points
        k_uniform = np.linspace(k.min(), k.max(), self.num_pixels)

        self._wavelength_to_k_map = k_uniform
        self._k_original = k

        # Initialize window function
        self._set_window(self._config.window_type)

        # Default dispersion coefficients
        self._dispersion_coeffs = np.array([0.0, 0.0, 0.0, 0.0])

    def _set_window(self, window_type: str):
        """Set spectral window function."""
        n = self.num_pixels
        if window_type == "hanning":
            self._window = np.hanning(n)
        elif window_type == "hamming":
            self._window = np.hamming(n)
        elif window_type == "gaussian":
            self._window = scipy_signal.windows.gaussian(n, std=n / 6)
        elif window_type == "rectangular":
            self._window = np.ones(n)
        else:
            self._window = np.hanning(n)

    def process_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """Process single spectrum to A-scan."""
        # Background subtraction
        if self._config.background_subtraction and self._background_spectrum is not None:
            spectrum = spectrum - self._background_spectrum

        # Resample to uniform k-space
        spectrum_k = self._resample_to_k_space(spectrum)

        # Apply dispersion compensation
        if self._config.apply_dispersion_compensation:
            spectrum_k = self._apply_dispersion_compensation(spectrum_k)

        # Apply spectral shaping window
        if self._config.apply_spectral_shaping:
            spectrum_k = spectrum_k * self._window

        # Zero padding
        padded_length = len(spectrum_k) * self._config.zero_padding_factor
        spectrum_padded = np.zeros(padded_length, dtype=complex)
        spectrum_padded[: len(spectrum_k)] = spectrum_k

        # FFT to get A-scan
        a_scan = np.fft.fft(spectrum_padded)

        # Take magnitude and use positive half
        a_scan = np.abs(a_scan[: padded_length // 2])

        return a_scan

    def _resample_to_k_space(self, spectrum: np.ndarray) -> np.ndarray:
        """Resample spectrum from wavelength to uniform k-space."""
        # Interpolation from λ to uniform k
        interpolator = interp1d(
            self._k_original,
            spectrum,
            kind="cubic",
            fill_value="extrapolate",
        )
        return interpolator(self._wavelength_to_k_map)

    def _apply_dispersion_compensation(self, spectrum: np.ndarray) -> np.ndarray:
        """Apply dispersion compensation."""
        if self._dispersion_coeffs is None:
            return spectrum

        k = self._wavelength_to_k_map
        k_center = k.mean()
        dk = k - k_center

        # Calculate phase correction
        phase = np.zeros_like(dk)
        for i, coeff in enumerate(self._dispersion_coeffs):
            phase += coeff * dk ** (i + 1)

        # Apply phase correction
        return spectrum * np.exp(-1j * phase)

    def process_b_scan(self, spectra: np.ndarray) -> np.ndarray:
        """Process multiple spectra to B-scan."""
        num_spectra = spectra.shape[1] if spectra.ndim > 1 else 1

        if num_spectra == 1:
            return self.process_spectrum(spectra.flatten())

        # Process each spectrum
        a_scan_length = self.num_pixels * self._config.zero_padding_factor // 2
        b_scan = np.zeros((a_scan_length, num_spectra))

        for i in range(num_spectra):
            b_scan[:, i] = self.process_spectrum(spectra[:, i])

        return b_scan

    def set_background(self, background: np.ndarray):
        """Set background spectrum for subtraction."""
        self._background_spectrum = background.copy()

    def set_dispersion_coefficients(self, coefficients: np.ndarray):
        """Set dispersion compensation coefficients."""
        self._dispersion_coeffs = coefficients

    def estimate_dispersion(
        self,
        reference_spectrum: np.ndarray,
        sample_spectrum: np.ndarray,
    ) -> np.ndarray:
        """Estimate dispersion from reference and sample spectra."""
        # Cross-correlation based dispersion estimation
        ref_k = self._resample_to_k_space(reference_spectrum)
        sample_k = self._resample_to_k_space(sample_spectrum)

        # Calculate phase difference
        phase_diff = np.angle(sample_k / (ref_k + 1e-10))

        # Unwrap phase
        phase_diff = np.unwrap(phase_diff)

        # Fit polynomial to get dispersion coefficients
        k = self._wavelength_to_k_map
        k_center = k.mean()
        dk = k - k_center

        # Fit 3rd order polynomial
        coeffs = np.polyfit(dk, phase_diff, 3)

        return coeffs[::-1]  # Return in order [a1, a2, a3, a4]


class OCTProcessor:
    """High-level OCT data processor."""

    def __init__(
        self,
        num_pixels: int = 2048,
        wavelength_range: tuple[float, float] = (800, 880),
    ):
        self._spectral_processor = SpectralProcessor(num_pixels, wavelength_range)
        self._output_bit_depth = 8

    def process_raw_data(
        self,
        raw_data: np.ndarray,
        log_scale: bool = True,
        dynamic_range: float = 50.0,
    ) -> np.ndarray:
        """Process raw OCT data to display image."""
        # Process spectral data
        if raw_data.ndim == 1:
            processed = self._spectral_processor.process_spectrum(raw_data)
        elif raw_data.ndim == 2:
            processed = self._spectral_processor.process_b_scan(raw_data)
        else:
            # 3D volume
            processed = self._process_volume(raw_data)

        # Log compression
        if log_scale:
            processed = 20 * np.log10(processed + 1e-10)

        # Apply dynamic range
        max_val = processed.max()
        min_val = max_val - dynamic_range
        processed = np.clip(processed, min_val, max_val)

        # Normalize to output bit depth
        processed = (processed - min_val) / dynamic_range
        processed = (processed * (2**self._output_bit_depth - 1)).astype(np.uint8)

        return processed

    def _process_volume(self, volume: np.ndarray) -> np.ndarray:
        """Process 3D volume data."""
        num_b_scans = volume.shape[2]
        result = []

        for i in range(num_b_scans):
            b_scan = self._spectral_processor.process_b_scan(volume[:, :, i])
            result.append(b_scan)

        return np.stack(result, axis=2)

    def segment_layers(
        self,
        b_scan: np.ndarray,
        num_layers: int = 3,
    ) -> list[np.ndarray]:
        """Simple layer segmentation based on intensity peaks."""
        # Find intensity peaks along each A-scan
        layers = []

        for layer_idx in range(num_layers):
            layer_positions = np.zeros(b_scan.shape[1])

            for i in range(b_scan.shape[1]):
                a_scan = b_scan[:, i]

                # Find peaks
                peaks, properties = scipy_signal.find_peaks(
                    a_scan,
                    height=a_scan.max() * 0.1,
                    distance=20,
                )

                if len(peaks) > layer_idx:
                    layer_positions[i] = peaks[layer_idx]
                elif len(peaks) > 0:
                    layer_positions[i] = peaks[-1]

            layers.append(layer_positions)

        return layers

    def calculate_thickness_map(
        self,
        layers: list[np.ndarray],
        pixel_size: float = 3.0,  # μm per pixel
    ) -> np.ndarray:
        """Calculate thickness between layers."""
        if len(layers) < 2:
            raise ValueError("Need at least 2 layers for thickness calculation")

        # Calculate thickness between first and last layer
        thickness = (layers[-1] - layers[0]) * pixel_size
        return thickness

    def apply_speckle_reduction(
        self,
        image: np.ndarray,
        method: str = "median",
        kernel_size: int = 3,
    ) -> np.ndarray:
        """Apply speckle reduction to OCT image."""
        from scipy.ndimage import median_filter, gaussian_filter

        if method == "median":
            return median_filter(image, size=kernel_size)
        elif method == "gaussian":
            return gaussian_filter(image, sigma=kernel_size / 2)
        elif method == "bilateral":
            # Simplified bilateral-like filtering
            smoothed = gaussian_filter(image.astype(float), sigma=kernel_size / 2)
            edges = np.abs(image.astype(float) - smoothed)
            weight = np.exp(-edges / edges.std())
            return (image * (1 - weight) + smoothed * weight).astype(image.dtype)
        else:
            return image

    def set_background(self, background: np.ndarray):
        """Set background spectrum."""
        self._spectral_processor.set_background(background)

    def set_dispersion_compensation(self, coefficients: np.ndarray):
        """Set dispersion compensation coefficients."""
        self._spectral_processor.set_dispersion_coefficients(coefficients)
