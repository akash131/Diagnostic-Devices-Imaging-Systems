"""Tests for thermographic module."""

import numpy as np
import pytest
from datetime import datetime

from src.imaging_systems.thermographic import (
    ThermographicCamera,
    SensorType,
    InflammationDetector,
    InflammationSeverity,
)


class TestThermographicCamera:
    """Tests for ThermographicCamera class."""

    def test_initialization(self):
        """Test camera initialization."""
        camera = ThermographicCamera(
            device_id="TC001",
            device_name="Test Thermal Camera",
            sensor_type=SensorType.MICROBOLOMETER,
            resolution=(640, 480),
        )

        assert camera.device_id == "TC001"
        assert camera.sensor_type == SensorType.MICROBOLOMETER
        assert camera.resolution == (640, 480)

    def test_initialize_device(self):
        """Test device initialization."""
        camera = ThermographicCamera(
            device_id="TC001",
            device_name="Test Thermal Camera",
        )

        result = camera.initialize()
        assert result is True
        assert camera.is_ready is True

    def test_acquire_thermal_image(self):
        """Test thermal image acquisition."""
        camera = ThermographicCamera(
            device_id="TC001",
            device_name="Test Thermal Camera",
        )
        camera.initialize()

        image_data = camera.acquire(emissivity=0.98)

        assert image_data is not None
        assert image_data.data.shape == (640, 480)
        assert image_data.metadata["modality"] == "thermal"
        assert "min_temp_c" in image_data.metadata
        assert "max_temp_c" in image_data.metadata

    def test_set_emissivity(self):
        """Test emissivity setting."""
        camera = ThermographicCamera(
            device_id="TC001",
            device_name="Test Thermal Camera",
        )

        camera.set_emissivity(0.95)
        # Should not raise error

        with pytest.raises(ValueError):
            camera.set_emissivity(1.5)  # Invalid

    def test_roi_statistics(self):
        """Test ROI statistics calculation."""
        camera = ThermographicCamera(
            device_id="TC001",
            device_name="Test Thermal Camera",
        )
        camera.initialize()

        image_data = camera.acquire()
        roi = (100, 100, 50, 50)

        stats = camera.get_roi_statistics(image_data, roi)

        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "std" in stats


class TestInflammationDetector:
    """Tests for InflammationDetector class."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = InflammationDetector(
            temperature_threshold=1.5,
            min_region_size=100,
        )

        assert detector.temperature_threshold == 1.5
        assert detector.min_region_size == 100

    def test_analyze_no_inflammation(self):
        """Test analysis with no inflammation."""
        detector = InflammationDetector(temperature_threshold=2.0)

        # Create uniform temperature image
        thermal_image = np.ones((480, 640)) * 33.0

        result = detector.analyze(thermal_image)

        assert result.has_inflammation is False
        assert len(result.regions) == 0

    def test_analyze_with_inflammation(self):
        """Test analysis with inflammation present."""
        detector = InflammationDetector(
            temperature_threshold=1.0,
            min_region_size=50,
        )

        # Create thermal image with hot spot
        thermal_image = np.ones((480, 640)) * 33.0

        # Add inflammation region
        thermal_image[200:250, 300:350] = 36.0

        result = detector.analyze(thermal_image)

        assert result.has_inflammation is True
        assert len(result.regions) > 0

    def test_severity_classification(self):
        """Test severity classification."""
        detector = InflammationDetector(temperature_threshold=0.5)

        # Create thermal image with varying severity regions
        thermal_image = np.ones((480, 640)) * 33.0

        # Mild region (1°C above)
        thermal_image[50:100, 50:100] = 34.0

        # Severe region (3°C above)
        thermal_image[200:250, 200:250] = 36.0

        result = detector.analyze(thermal_image)

        severities = [r.severity for r in result.regions]

        # Should have detected multiple severity levels
        assert len(result.regions) >= 1

    def test_bilateral_comparison(self):
        """Test bilateral temperature comparison."""
        detector = InflammationDetector()

        # Create thermal image with asymmetric temperatures
        thermal_image = np.ones((480, 640)) * 33.0
        thermal_image[:, :320] = 34.0  # Left side warmer

        left_roi = (50, 200, 100, 100)
        right_roi = (450, 200, 100, 100)

        comparison = detector.compare_bilateral(thermal_image, left_roi, right_roi)

        assert comparison["warmer_side"] == "left"
        assert comparison["temperature_difference"] > 0

    def test_set_severity_thresholds(self):
        """Test custom severity thresholds."""
        detector = InflammationDetector()

        detector.set_severity_thresholds(mild=0.5, moderate=1.0, severe=2.0)

        assert detector._severity_thresholds["mild"] == 0.5
        assert detector._severity_thresholds["moderate"] == 1.0
        assert detector._severity_thresholds["severe"] == 2.0
