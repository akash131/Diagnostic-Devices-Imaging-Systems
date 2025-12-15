"""DICOM parsing and validation utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, BinaryIO
import struct
import numpy as np

from .dataset import DicomDataset
from .tags import DicomTag, DicomTags, VR


@dataclass
class ValidationResult:
    """Result of DICOM validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]

    def add_error(self, message: str):
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str):
        self.warnings.append(message)


class DicomParser:
    """Parser for DICOM files."""

    DICOM_PREAMBLE_SIZE = 128
    DICOM_PREFIX = b"DICM"

    def __init__(self):
        self._vr_lengths = {
            "AE": 16, "AS": 4, "AT": 4, "CS": 16, "DA": 8, "DS": 16,
            "DT": 26, "FL": 4, "FD": 8, "IS": 12, "LO": 64, "LT": 10240,
            "OB": None, "OD": None, "OF": None, "OW": None, "PN": 64,
            "SH": 16, "SL": 4, "SQ": None, "SS": 2, "ST": 1024,
            "TM": 16, "UI": 64, "UL": 4, "UN": None, "US": 2, "UT": None,
        }

    def parse_file(self, file_path: str | Path) -> DicomDataset:
        """Parse a DICOM file."""
        with open(file_path, "rb") as f:
            return self._parse_stream(f)

    def parse_bytes(self, data: bytes) -> DicomDataset:
        """Parse DICOM data from bytes."""
        import io
        return self._parse_stream(io.BytesIO(data))

    def _parse_stream(self, stream: BinaryIO) -> DicomDataset:
        """Parse DICOM from a stream."""
        dataset = DicomDataset()

        # Skip preamble
        stream.seek(self.DICOM_PREAMBLE_SIZE)

        # Check DICM prefix
        prefix = stream.read(4)
        if prefix != self.DICOM_PREFIX:
            # Try without preamble (old DICOM format)
            stream.seek(0)

        # Read elements
        while True:
            element = self._read_element(stream)
            if element is None:
                break

            tag, vr, value = element

            # Handle pixel data specially
            if tag == (0x7FE0, 0x0010):
                dataset._pixel_data = self._parse_pixel_data(value, dataset)
            else:
                # Create DicomTag and add to dataset
                dicom_tag = self._find_tag(tag)
                if dicom_tag:
                    dataset.add_element(dicom_tag, value)

        return dataset

    def _read_element(self, stream: BinaryIO) -> Optional[tuple]:
        """Read a single DICOM element."""
        # Read tag
        tag_bytes = stream.read(4)
        if len(tag_bytes) < 4:
            return None

        group, element = struct.unpack("<HH", tag_bytes)
        tag = (group, element)

        # Read VR (2 bytes for explicit VR)
        vr_bytes = stream.read(2)
        if len(vr_bytes) < 2:
            return None

        vr = vr_bytes.decode("ascii", errors="ignore")

        # Determine value length
        if vr in ("OB", "OD", "OF", "OL", "OW", "SQ", "UC", "UN", "UR", "UT"):
            # Skip 2 reserved bytes, then read 4-byte length
            stream.read(2)
            length_bytes = stream.read(4)
            if len(length_bytes) < 4:
                return None
            length = struct.unpack("<I", length_bytes)[0]
        else:
            # Read 2-byte length
            length_bytes = stream.read(2)
            if len(length_bytes) < 2:
                return None
            length = struct.unpack("<H", length_bytes)[0]

        # Handle undefined length
        if length == 0xFFFFFFFF:
            value = self._read_undefined_length(stream)
        elif length > 0:
            value = stream.read(length)
        else:
            value = b""

        # Parse value based on VR
        parsed_value = self._parse_value(vr, value)

        return tag, vr, parsed_value

    def _read_undefined_length(self, stream: BinaryIO) -> bytes:
        """Read data with undefined length until sequence delimiter."""
        data = []
        while True:
            tag_bytes = stream.read(4)
            if len(tag_bytes) < 4:
                break

            group, element = struct.unpack("<HH", tag_bytes)

            # Check for sequence delimitation item
            if group == 0xFFFE and element == 0xE0DD:
                stream.read(4)  # Skip length (should be 0)
                break

            data.append(tag_bytes)
            # Read the rest of the item
            length_bytes = stream.read(4)
            data.append(length_bytes)
            length = struct.unpack("<I", length_bytes)[0]
            if length != 0xFFFFFFFF and length > 0:
                data.append(stream.read(length))

        return b"".join(data)

    def _parse_value(self, vr: str, value: bytes) -> any:
        """Parse value based on VR."""
        if not value:
            return None

        try:
            if vr in ("AE", "AS", "CS", "DA", "DS", "DT", "IS", "LO", "LT",
                      "PN", "SH", "ST", "TM", "UI", "UT"):
                return value.decode("ascii", errors="ignore").strip().strip("\x00")
            elif vr == "US":
                return struct.unpack("<H", value[:2])[0]
            elif vr == "SS":
                return struct.unpack("<h", value[:2])[0]
            elif vr == "UL":
                return struct.unpack("<I", value[:4])[0]
            elif vr == "SL":
                return struct.unpack("<i", value[:4])[0]
            elif vr == "FL":
                return struct.unpack("<f", value[:4])[0]
            elif vr == "FD":
                return struct.unpack("<d", value[:8])[0]
            elif vr in ("OB", "OW", "OD", "OF", "UN"):
                return value
            else:
                return value
        except Exception:
            return value

    def _parse_pixel_data(self, data: bytes, dataset: DicomDataset) -> np.ndarray:
        """Parse pixel data into numpy array."""
        rows = dataset.get_value(DicomTags.Rows, 0)
        cols = dataset.get_value(DicomTags.Columns, 0)
        bits_allocated = dataset.get_value(DicomTags.BitsAllocated, 16)
        samples_per_pixel = dataset.get_value(DicomTags.SamplesPerPixel, 1)
        pixel_rep = dataset.get_value(DicomTags.PixelRepresentation, 0)

        if rows == 0 or cols == 0:
            return np.frombuffer(data, dtype=np.uint8)

        # Determine dtype
        if bits_allocated == 8:
            dtype = np.uint8
        elif bits_allocated == 16:
            dtype = np.int16 if pixel_rep == 1 else np.uint16
        elif bits_allocated == 32:
            dtype = np.float32
        else:
            dtype = np.uint16

        # Create array
        try:
            arr = np.frombuffer(data, dtype=dtype)
            if samples_per_pixel == 1:
                arr = arr.reshape((rows, cols))
            else:
                arr = arr.reshape((rows, cols, samples_per_pixel))
            return arr
        except ValueError:
            return np.frombuffer(data, dtype=dtype)

    def _find_tag(self, tag: tuple[int, int]) -> Optional[DicomTag]:
        """Find DicomTag by group and element."""
        for attr_name in dir(DicomTags):
            attr = getattr(DicomTags, attr_name)
            if isinstance(attr, DicomTag) and attr.tag == tag:
                return attr
        return None


class DicomValidator:
    """Validator for DICOM datasets."""

    # Required tags by IOD
    REQUIRED_PATIENT_TAGS = [
        DicomTags.PatientName,
        DicomTags.PatientID,
    ]

    REQUIRED_STUDY_TAGS = [
        DicomTags.StudyInstanceUID,
        DicomTags.StudyDate,
    ]

    REQUIRED_SERIES_TAGS = [
        DicomTags.SeriesInstanceUID,
        DicomTags.Modality,
    ]

    REQUIRED_IMAGE_TAGS = [
        DicomTags.SOPClassUID,
        DicomTags.SOPInstanceUID,
        DicomTags.Rows,
        DicomTags.Columns,
        DicomTags.BitsAllocated,
        DicomTags.BitsStored,
        DicomTags.HighBit,
        DicomTags.PixelRepresentation,
        DicomTags.SamplesPerPixel,
        DicomTags.PhotometricInterpretation,
    ]

    def __init__(self, strict: bool = False):
        self.strict = strict

    def validate(self, dataset: DicomDataset) -> ValidationResult:
        """Validate a DICOM dataset."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])

        # Check required tags
        self._check_required_tags(dataset, self.REQUIRED_PATIENT_TAGS, "Patient", result)
        self._check_required_tags(dataset, self.REQUIRED_STUDY_TAGS, "Study", result)
        self._check_required_tags(dataset, self.REQUIRED_SERIES_TAGS, "Series", result)
        self._check_required_tags(dataset, self.REQUIRED_IMAGE_TAGS, "Image", result)

        # Validate pixel data
        self._validate_pixel_data(dataset, result)

        # Validate UID format
        self._validate_uids(dataset, result)

        # Validate date/time formats
        self._validate_dates(dataset, result)

        return result

    def _check_required_tags(
        self,
        dataset: DicomDataset,
        tags: list[DicomTag],
        module_name: str,
        result: ValidationResult,
    ):
        """Check for required tags."""
        for tag in tags:
            if not dataset.has_element(tag):
                if self.strict:
                    result.add_error(f"Missing required {module_name} tag: {tag.keyword}")
                else:
                    result.add_warning(f"Missing recommended {module_name} tag: {tag.keyword}")

    def _validate_pixel_data(self, dataset: DicomDataset, result: ValidationResult):
        """Validate pixel data consistency."""
        if dataset.pixel_data is None:
            result.add_warning("No pixel data present")
            return

        rows = dataset.get_value(DicomTags.Rows)
        cols = dataset.get_value(DicomTags.Columns)

        if rows and cols:
            expected_shape = (rows, cols)
            actual_shape = dataset.pixel_data.shape[:2]

            if expected_shape != actual_shape:
                result.add_error(
                    f"Pixel data shape mismatch: expected {expected_shape}, got {actual_shape}"
                )

    def _validate_uids(self, dataset: DicomDataset, result: ValidationResult):
        """Validate UID formats."""
        uid_tags = [
            DicomTags.StudyInstanceUID,
            DicomTags.SeriesInstanceUID,
            DicomTags.SOPInstanceUID,
            DicomTags.SOPClassUID,
        ]

        for tag in uid_tags:
            uid = dataset.get_value(tag)
            if uid:
                if not self._is_valid_uid(uid):
                    result.add_error(f"Invalid UID format for {tag.keyword}: {uid}")

    def _is_valid_uid(self, uid: str) -> bool:
        """Check if UID format is valid."""
        if len(uid) > 64:
            return False
        if not uid:
            return False
        # UIDs should only contain digits and dots
        valid_chars = set("0123456789.")
        if not all(c in valid_chars for c in uid):
            return False
        # Should not start or end with dot
        if uid.startswith(".") or uid.endswith("."):
            return False
        # Should not have consecutive dots
        if ".." in uid:
            return False
        return True

    def _validate_dates(self, dataset: DicomDataset, result: ValidationResult):
        """Validate date formats."""
        date_tags = [
            DicomTags.StudyDate,
            DicomTags.SeriesDate,
            DicomTags.PatientBirthDate,
        ]

        for tag in date_tags:
            date_str = dataset.get_value(tag)
            if date_str:
                if not self._is_valid_date(date_str):
                    result.add_warning(f"Invalid date format for {tag.keyword}: {date_str}")

    def _is_valid_date(self, date_str: str) -> bool:
        """Check if date format is valid (YYYYMMDD)."""
        if len(date_str) != 8:
            return False
        try:
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            return 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31
        except ValueError:
            return False

    def validate_iod(self, dataset: DicomDataset, modality: str) -> ValidationResult:
        """Validate against specific IOD (Information Object Definition)."""
        result = self.validate(dataset)

        # Add modality-specific validation
        if modality == "US":
            self._validate_ultrasound_iod(dataset, result)
        elif modality == "DX":
            self._validate_xray_iod(dataset, result)
        elif modality == "OPT":
            self._validate_oct_iod(dataset, result)

        return result

    def _validate_ultrasound_iod(self, dataset: DicomDataset, result: ValidationResult):
        """Validate ultrasound-specific requirements."""
        from .tags import ModalityTags

        # Check for recommended US tags
        if not dataset.has_element(ModalityTags.US.TransducerFrequency):
            result.add_warning("Missing recommended Transducer Frequency")

    def _validate_xray_iod(self, dataset: DicomDataset, result: ValidationResult):
        """Validate X-ray specific requirements."""
        from .tags import ModalityTags

        # Check for recommended DX tags
        if not dataset.has_element(ModalityTags.DX.KVP):
            result.add_warning("Missing recommended KVP")

    def _validate_oct_iod(self, dataset: DicomDataset, result: ValidationResult):
        """Validate OCT-specific requirements."""
        # Add OCT-specific validations
        pass
