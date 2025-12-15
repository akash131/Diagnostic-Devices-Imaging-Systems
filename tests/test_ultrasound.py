"""Tests for ultrasound module."""

import numpy as np
import pytest

from src.imaging_systems.ultrasound import (
    UltrasoundTransducer,
    TransducerType,
    SignalProcessor,
    BeamFormer,
    ImageReconstructor,
)


class TestUltrasoundTransducer:
    """Tests for UltrasoundTransducer class."""

    def test_initialization(self):
        """Test transducer initialization."""
        transducer = UltrasoundTransducer(
            device_id="US001",
            device_name="Test Transducer",
            transducer_type=TransducerType.LINEAR,
            center_frequency=7.5,
        )

        assert transducer.device_id == "US001"
        assert transducer.transducer_type == TransducerType.LINEAR
        assert transducer.center_frequency == 7.5

    def test_initialize_device(self):
        """Test device initialization."""
        transducer = UltrasoundTransducer(
            device_id="US001",
            device_name="Test Transducer",
            transducer_type=TransducerType.LINEAR,
            center_frequency=7.5,
        )

        result = transducer.initialize()
        assert result is True
        assert transducer.is_ready is True

    def test_acquire_image(self):
        """Test image acquisition."""
        transducer = UltrasoundTransducer(
            device_id="US001",
            device_name="Test Transducer",
            transducer_type=TransducerType.LINEAR,
            center_frequency=7.5,
        )
        transducer.initialize()

        image_data = transducer.acquire(depth=100.0, gain=1.0)

        assert image_data is not None
        assert image_data.data.shape[0] > 0
        assert image_data.metadata["modality"] == "ultrasound"
        assert image_data.metadata["depth_mm"] == 100.0

    def test_calibration(self):
        """Test transducer calibration."""
        transducer = UltrasoundTransducer(
            device_id="US001",
            device_name="Test Transducer",
            transducer_type=TransducerType.LINEAR,
            center_frequency=7.5,
        )
        transducer.initialize()

        calibration = transducer.calibrate()

        assert calibration is not None
        assert calibration.is_valid is True
        assert "element_sensitivity" in calibration.calibration_parameters


class TestSignalProcessor:
    """Tests for SignalProcessor class."""

    def test_initialization(self):
        """Test signal processor initialization."""
        processor = SignalProcessor(
            processor_id="SP001",
            sampling_rate=40e6,
            center_frequency=5e6,
        )

        assert processor.processor_id == "SP001"
        assert processor.sampling_rate == 40e6

    def test_process_signal(self):
        """Test signal processing."""
        processor = SignalProcessor(
            processor_id="SP001",
            sampling_rate=40e6,
            center_frequency=5e6,
        )

        # Generate test RF signal
        t = np.arange(1000) / 40e6
        rf_signal = np.sin(2 * np.pi * 5e6 * t) * np.exp(-t * 1e5)

        processed = processor.process(rf_signal)

        assert processed is not None
        assert len(processed) > 0


class TestBeamFormer:
    """Tests for BeamFormer class."""

    def test_initialization(self):
        """Test beamformer initialization."""
        beamformer = BeamFormer(
            num_elements=128,
            element_pitch=0.3,
            center_frequency=5e6,
            sampling_rate=40e6,
        )

        assert beamformer.num_elements == 128
        assert beamformer.element_pitch == 0.3

    def test_delay_and_sum(self):
        """Test delay and sum beamforming."""
        beamformer = BeamFormer(
            num_elements=64,
            element_pitch=0.3,
            center_frequency=5e6,
            sampling_rate=40e6,
        )

        # Generate test RF data
        rf_data = np.random.randn(1000, 64)

        result = beamformer.delay_and_sum(rf_data, focal_point=(0, 50))

        assert result is not None
        assert len(result) > 0


class TestImageReconstructor:
    """Tests for ImageReconstructor class."""

    def test_initialization(self):
        """Test reconstructor initialization."""
        reconstructor = ImageReconstructor(output_width=512, output_height=512)

        assert reconstructor.output_width == 512
        assert reconstructor.output_height == 512

    def test_reconstruct_b_mode(self):
        """Test B-mode reconstruction."""
        reconstructor = ImageReconstructor(output_width=256, output_height=256)

        # Generate test scan lines
        scan_lines = np.random.rand(500, 128).astype(np.uint8) * 255

        result = reconstructor.reconstruct_b_mode(scan_lines, scan_geometry="linear")

        assert result is not None
        assert result.data.shape == (256, 256)
