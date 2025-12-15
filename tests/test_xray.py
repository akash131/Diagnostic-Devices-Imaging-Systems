"""Tests for X-ray module."""

import numpy as np
import pytest

from src.imaging_systems.xray import (
    PortableXRay,
    XRayDetector,
    DetectorType,
    ExposureParameters,
)


class TestPortableXRay:
    """Tests for PortableXRay class."""

    def test_initialization(self):
        """Test X-ray system initialization."""
        xray = PortableXRay(
            device_id="XR001",
            device_name="Test X-Ray",
            max_kvp=125,
            max_ma=320,
        )

        assert xray.device_id == "XR001"
        assert xray.battery_level == 100.0

    def test_initialize_device(self):
        """Test device initialization."""
        xray = PortableXRay(
            device_id="XR001",
            device_name="Test X-Ray",
        )

        result = xray.initialize()
        assert result is True
        assert xray.is_ready is True

    def test_acquire_image(self):
        """Test image acquisition."""
        xray = PortableXRay(
            device_id="XR001",
            device_name="Test X-Ray",
        )
        xray.initialize()

        image_data = xray.acquire(anatomical_region="chest")

        assert image_data is not None
        assert image_data.data.shape == (2048, 2048)
        assert image_data.metadata["modality"] == "xray"

    def test_custom_exposure_parameters(self):
        """Test custom exposure parameters."""
        xray = PortableXRay(
            device_id="XR001",
            device_name="Test X-Ray",
        )
        xray.initialize()

        params = ExposureParameters(
            kvp=70,
            ma=150,
            exposure_time=0.05,
            focal_spot="small",
            sfd=100,
        )

        image_data = xray.acquire(exposure_params=params)

        assert image_data.metadata["kvp"] == 70
        assert image_data.metadata["ma"] == 150

    def test_invalid_exposure_parameters(self):
        """Test invalid exposure parameters raise error."""
        xray = PortableXRay(
            device_id="XR001",
            device_name="Test X-Ray",
        )
        xray.initialize()

        params = ExposureParameters(
            kvp=200,  # Too high
            ma=150,
            exposure_time=0.05,
            focal_spot="small",
            sfd=100,
        )

        with pytest.raises(ValueError):
            xray.acquire(exposure_params=params)


class TestExposureParameters:
    """Tests for ExposureParameters class."""

    def test_mas_calculation(self):
        """Test mAs calculation."""
        params = ExposureParameters(
            kvp=80,
            ma=200,
            exposure_time=0.1,
            focal_spot="large",
            sfd=100,
        )

        assert params.mas == 20.0

    def test_validate_valid_params(self):
        """Test validation of valid parameters."""
        params = ExposureParameters(
            kvp=80,
            ma=200,
            exposure_time=0.05,
            focal_spot="large",
            sfd=100,
        )

        assert params.validate() is True

    def test_validate_invalid_kvp(self):
        """Test validation with invalid kVp."""
        params = ExposureParameters(
            kvp=200,  # Invalid
            ma=200,
            exposure_time=0.05,
            focal_spot="large",
            sfd=100,
        )

        assert params.validate() is False


class TestXRayDetector:
    """Tests for XRayDetector class."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = XRayDetector(
            detector_id="DET001",
            detector_type=DetectorType.FLAT_PANEL_INDIRECT,
        )

        assert detector.detector_id == "DET001"
        assert detector.detector_type == DetectorType.FLAT_PANEL_INDIRECT

    def test_initialize(self):
        """Test detector initialization."""
        detector = XRayDetector(
            detector_id="DET001",
            detector_type=DetectorType.FLAT_PANEL_INDIRECT,
        )

        result = detector.initialize()
        assert result is True

    def test_calibrate(self):
        """Test detector calibration."""
        detector = XRayDetector(
            detector_id="DET001",
            detector_type=DetectorType.FLAT_PANEL_INDIRECT,
        )
        detector.initialize()

        result = detector.calibrate()
        assert result is True
