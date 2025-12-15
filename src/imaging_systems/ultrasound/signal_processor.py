"""Ultrasound signal processing implementation."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import signal as scipy_signal

from ..base import SignalProcessorBase, ImageData


@dataclass
class ProcessingParameters:
    """Parameters for ultrasound signal processing."""

    sampling_frequency: float  # MHz
    center_frequency: float  # MHz
    dynamic_range: float  # dB
    persistence: int  # frames
    edge_enhancement: float  # 0-1
    speckle_reduction: float  # 0-1


class SignalProcessor(SignalProcessorBase):
    """Ultrasound signal processor for RF data."""

    def __init__(
        self,
        processor_id: str,
        sampling_rate: float = 40e6,
        center_frequency: float = 5e6,
    ):
        super().__init__(processor_id, sampling_rate)
        self.center_frequency = center_frequency
        self._params = ProcessingParameters(
            sampling_frequency=sampling_rate / 1e6,
            center_frequency=center_frequency / 1e6,
            dynamic_range=60.0,
            persistence=2,
            edge_enhancement=0.3,
            speckle_reduction=0.5,
        )

    def process(self, signal: np.ndarray) -> np.ndarray:
        """Process RF signal through the complete pipeline."""
        # Apply any custom filters first
        filtered = self.apply_filters(signal)

        # Bandpass filter
        filtered = self._bandpass_filter(filtered)

        # Envelope detection (Hilbert transform)
        envelope = self._envelope_detection(filtered)

        # Log compression
        compressed = self._log_compress(envelope)

        return compressed

    def _bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """Apply bandpass filter centered on transducer frequency."""
        fs = self.sampling_rate
        low = self.center_frequency * 0.5
        high = self.center_frequency * 1.5

        # Ensure valid frequency range
        nyq = fs / 2
        low_norm = max(low / nyq, 0.01)
        high_norm = min(high / nyq, 0.99)

        b, a = scipy_signal.butter(4, [low_norm, high_norm], btype="band")
        return scipy_signal.filtfilt(b, a, signal, axis=0)

    def _envelope_detection(self, signal: np.ndarray) -> np.ndarray:
        """Extract signal envelope using Hilbert transform."""
        analytic = scipy_signal.hilbert(signal, axis=0)
        return np.abs(analytic)

    def _log_compress(self, signal: np.ndarray) -> np.ndarray:
        """Apply logarithmic compression for display."""
        # Avoid log of zero
        signal = np.maximum(signal, 1e-10)

        # Log compression
        log_signal = 20 * np.log10(signal)

        # Apply dynamic range
        max_val = log_signal.max()
        min_val = max_val - self._params.dynamic_range
        compressed = np.clip(log_signal, min_val, max_val)

        # Normalize to 0-255 for display
        normalized = 255 * (compressed - min_val) / self._params.dynamic_range
        return normalized.astype(np.uint8)

    def set_parameters(self, **params):
        """Update processing parameters."""
        for key, value in params.items():
            if hasattr(self._params, key):
                setattr(self._params, key, value)


class BeamFormer:
    """Beamforming processor for ultrasound data."""

    def __init__(
        self,
        num_elements: int,
        element_pitch: float,
        center_frequency: float,
        sampling_rate: float,
    ):
        self.num_elements = num_elements
        self.element_pitch = element_pitch
        self.center_frequency = center_frequency
        self.sampling_rate = sampling_rate
        self.speed_of_sound = 1540.0  # m/s in tissue

        self._apodization = np.hanning(num_elements)

    def delay_and_sum(
        self,
        rf_data: np.ndarray,
        focal_point: tuple[float, float],
    ) -> np.ndarray:
        """Apply delay-and-sum beamforming."""
        num_samples, num_channels = rf_data.shape
        fx, fz = focal_point

        # Calculate element positions
        element_positions = np.arange(num_channels) * self.element_pitch
        element_positions -= element_positions.mean()

        # Calculate delays
        distances = np.sqrt((element_positions - fx) ** 2 + fz**2)
        delays = distances / self.speed_of_sound

        # Convert delays to sample indices
        delay_samples = (delays * self.sampling_rate).astype(int)
        delay_samples -= delay_samples.min()

        # Apply delays and sum
        max_delay = delay_samples.max()
        output_length = num_samples - max_delay

        output = np.zeros(output_length)
        for ch in range(num_channels):
            start = delay_samples[ch]
            end = start + output_length
            output += rf_data[start:end, ch] * self._apodization[ch]

        return output / num_channels

    def dynamic_focusing(
        self,
        rf_data: np.ndarray,
        depth_range: tuple[float, float],
        num_focal_points: int = 100,
    ) -> np.ndarray:
        """Apply dynamic receive focusing."""
        num_samples, num_channels = rf_data.shape
        depths = np.linspace(depth_range[0], depth_range[1], num_focal_points)

        # Element positions
        element_positions = np.arange(num_channels) * self.element_pitch
        element_positions -= element_positions.mean()

        # Calculate sample indices for each depth
        sample_rate_mm = self.sampling_rate * 1000 / self.speed_of_sound / 2

        output = np.zeros((num_focal_points, num_channels))

        for i, depth in enumerate(depths):
            # Calculate travel times for each element
            distances = np.sqrt(element_positions**2 + depth**2)
            sample_indices = (distances * sample_rate_mm).astype(int)

            # Extract samples at calculated indices
            for ch in range(num_channels):
                idx = sample_indices[ch]
                if 0 <= idx < num_samples:
                    output[i, ch] = rf_data[idx, ch] * self._apodization[ch]

        return output.sum(axis=1)

    def set_apodization(self, window_type: str = "hanning"):
        """Set apodization window type."""
        n = self.num_elements
        if window_type == "hanning":
            self._apodization = np.hanning(n)
        elif window_type == "hamming":
            self._apodization = np.hamming(n)
        elif window_type == "blackman":
            self._apodization = np.blackman(n)
        elif window_type == "rectangular":
            self._apodization = np.ones(n)
        else:
            raise ValueError(f"Unknown window type: {window_type}")


class ImageReconstructor:
    """Reconstructs ultrasound images from processed data."""

    def __init__(
        self,
        output_width: int = 512,
        output_height: int = 512,
    ):
        self.output_width = output_width
        self.output_height = output_height

    def reconstruct_b_mode(
        self,
        scan_lines: np.ndarray,
        scan_geometry: str = "linear",
    ) -> ImageData:
        """Reconstruct B-mode image from scan lines."""
        num_samples, num_lines = scan_lines.shape

        if scan_geometry == "linear":
            image = self._reconstruct_linear(scan_lines)
        elif scan_geometry == "sector":
            image = self._reconstruct_sector(scan_lines)
        elif scan_geometry == "convex":
            image = self._reconstruct_convex(scan_lines)
        else:
            raise ValueError(f"Unknown scan geometry: {scan_geometry}")

        return ImageData(
            data=image,
            metadata={
                "reconstruction_type": "b_mode",
                "scan_geometry": scan_geometry,
                "output_size": (self.output_width, self.output_height),
            },
            modality="ultrasound",
        )

    def _reconstruct_linear(self, scan_lines: np.ndarray) -> np.ndarray:
        """Reconstruct linear scan geometry."""
        from scipy.ndimage import zoom

        # Simple bilinear interpolation to output size
        zoom_factors = (
            self.output_height / scan_lines.shape[0],
            self.output_width / scan_lines.shape[1],
        )
        return zoom(scan_lines, zoom_factors, order=1)

    def _reconstruct_sector(self, scan_lines: np.ndarray) -> np.ndarray:
        """Reconstruct sector scan geometry."""
        num_samples, num_lines = scan_lines.shape
        output = np.zeros((self.output_height, self.output_width))

        # Sector angle range (typically ±45 degrees)
        angles = np.linspace(-np.pi / 4, np.pi / 4, num_lines)

        # Map polar to Cartesian coordinates
        center_x = self.output_width // 2
        center_y = 0

        for i, angle in enumerate(angles):
            for j in range(num_samples):
                r = j * self.output_height / num_samples
                x = int(center_x + r * np.sin(angle))
                y = int(center_y + r * np.cos(angle))

                if 0 <= x < self.output_width and 0 <= y < self.output_height:
                    output[y, x] = scan_lines[j, i]

        return output

    def _reconstruct_convex(self, scan_lines: np.ndarray) -> np.ndarray:
        """Reconstruct convex scan geometry."""
        # Similar to sector but with curved array origin
        num_samples, num_lines = scan_lines.shape
        output = np.zeros((self.output_height, self.output_width))

        # Convex array parameters
        array_radius = 60  # mm (typical convex array radius)
        angles = np.linspace(-np.pi / 3, np.pi / 3, num_lines)

        center_x = self.output_width // 2
        center_y = -int(array_radius * self.output_height / 200)

        for i, angle in enumerate(angles):
            for j in range(num_samples):
                r = array_radius + j * 150 / num_samples
                x = int(center_x + r * np.sin(angle))
                y = int(center_y + r * np.cos(angle))

                if 0 <= x < self.output_width and 0 <= y < self.output_height:
                    output[y, x] = scan_lines[j, i]

        return output

    def apply_post_processing(
        self,
        image: np.ndarray,
        speckle_reduction: float = 0.5,
        edge_enhancement: float = 0.3,
    ) -> np.ndarray:
        """Apply post-processing filters to the image."""
        from scipy.ndimage import median_filter, gaussian_laplace

        # Speckle reduction using median filter
        if speckle_reduction > 0:
            kernel_size = int(3 + speckle_reduction * 4)
            if kernel_size % 2 == 0:
                kernel_size += 1
            image = median_filter(image, size=kernel_size)

        # Edge enhancement using unsharp masking
        if edge_enhancement > 0:
            laplacian = gaussian_laplace(image.astype(float), sigma=1)
            image = image - edge_enhancement * laplacian

        return np.clip(image, 0, 255).astype(np.uint8)
