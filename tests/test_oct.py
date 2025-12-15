"""Tests for OCT module."""

import numpy as np
import pytest

from src.imaging_systems.oct import (
    OCTDevice,
    OCTType,
    ScanPattern,
    OCTScanner,
    OCTProcessor,
)


class TestOCTDevice:
    """Tests for OCTDevice class."""

    def test_initialization(self):
        """Test OCT device initialization."""
        oct = OCTDevice(
            device_id="OCT001",
            device_name="Test OCT",
            oct_type=OCTType.SPECTRAL_DOMAIN,
            center_wavelength=840.0,
            bandwidth=50.0,
        )

        assert oct.device_id == "OCT001"
        assert oct.oct_type == OCTType.SPECTRAL_DOMAIN
        assert oct.center_wavelength == 840.0

    def test_initialize_device(self):
        """Test device initialization."""
        oct = OCTDevice(
            device_id="OCT001",
            device_name="Test OCT",
            oct_type=OCTType.SPECTRAL_DOMAIN,
        )

        result = oct.initialize()
        assert result is True
        assert oct.is_ready is True

    def test_acquire_b_scan(self):
        """Test B-scan acquisition."""
        oct = OCTDevice(
            device_id="OCT001",
            device_name="Test OCT",
        )
        oct.initialize()

        image_data = oct.acquire(
            scan_pattern=ScanPattern.B_SCAN,
            num_a_scans=512,
        )

        assert image_data is not None
        assert image_data.metadata["modality"] == "oct"
        assert image_data.metadata["scan_pattern"] == "b_scan"

    def test_acquire_volume(self):
        """Test volume acquisition."""
        oct = OCTDevice(
            device_id="OCT001",
            device_name="Test OCT",
        )
        oct.initialize()

        image_data = oct.acquire(
            scan_pattern=ScanPattern.VOLUME,
            num_a_scans=128,
            num_b_scans=64,
        )

        assert image_data is not None
        assert image_data.data.ndim == 3

    def test_specs_calculation(self):
        """Test specifications calculation."""
        oct = OCTDevice(
            device_id="OCT001",
            device_name="Test OCT",
            oct_type=OCTType.SPECTRAL_DOMAIN,
            center_wavelength=840.0,
            bandwidth=50.0,
        )

        specs = oct.specs
        assert specs.center_wavelength == 840.0
        assert specs.axial_resolution > 0


class TestOCTScanner:
    """Tests for OCTScanner class."""

    def test_initialization(self):
        """Test scanner initialization."""
        scanner = OCTScanner(
            galvo_x_sensitivity=1.0,
            galvo_y_sensitivity=1.0,
        )

        result = scanner.initialize()
        assert result is True

    def test_set_scan_pattern_linear(self):
        """Test linear scan pattern generation."""
        scanner = OCTScanner()
        scanner.initialize()

        x_wave, y_wave = scanner.set_scan_pattern(
            "linear",
            width=6.0,
            num_a_scans=512,
        )

        assert len(x_wave) > 0
        assert len(y_wave) > 0

    def test_set_scan_pattern_circular(self):
        """Test circular scan pattern generation."""
        scanner = OCTScanner()
        scanner.initialize()

        x_wave, y_wave = scanner.set_scan_pattern(
            "circular",
            diameter=3.4,
            num_points=768,
        )

        assert len(x_wave) == 768
        assert len(y_wave) == 768


class TestOCTProcessor:
    """Tests for OCTProcessor class."""

    def test_initialization(self):
        """Test processor initialization."""
        processor = OCTProcessor(
            num_pixels=2048,
            wavelength_range=(800, 880),
        )

        assert processor is not None

    def test_process_raw_data(self):
        """Test raw data processing."""
        processor = OCTProcessor(num_pixels=1024)

        # Generate test spectral data
        raw_data = np.random.rand(1024, 128)

        processed = processor.process_raw_data(raw_data, log_scale=True)

        assert processed is not None
        assert processed.dtype == np.uint8

    def test_speckle_reduction(self):
        """Test speckle reduction."""
        processor = OCTProcessor()

        # Generate test image
        image = (np.random.rand(256, 256) * 255).astype(np.uint8)

        filtered = processor.apply_speckle_reduction(image, method="median")

        assert filtered.shape == image.shape
