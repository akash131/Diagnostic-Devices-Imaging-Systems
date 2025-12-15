"""DICOM dataset classes for managing medical imaging data."""

from dataclasses import dataclass, field
from datetime import datetime, date, time
from pathlib import Path
from typing import Any, Optional, Iterator
import numpy as np

from .tags import DicomTag, DicomTags, VR
from .uid import generate_uid


@dataclass
class DicomElement:
    """A single DICOM data element."""

    tag: DicomTag
    value: Any

    @property
    def group(self) -> int:
        return self.tag.group

    @property
    def element(self) -> int:
        return self.tag.element

    def __repr__(self):
        return f"DicomElement({self.tag.tag_hex} {self.tag.keyword}: {self.value})"


class DicomDataset:
    """Container for DICOM data elements."""

    def __init__(self):
        self._elements: dict[tuple[int, int], DicomElement] = {}
        self._pixel_data: Optional[np.ndarray] = None
        self._file_path: Optional[Path] = None

    def add_element(self, tag: DicomTag, value: Any):
        """Add a data element to the dataset."""
        self._elements[tag.tag] = DicomElement(tag, value)

    def get_element(self, tag: DicomTag) -> Optional[DicomElement]:
        """Get a data element by tag."""
        return self._elements.get(tag.tag)

    def get_value(self, tag: DicomTag, default: Any = None) -> Any:
        """Get value of a data element."""
        element = self._elements.get(tag.tag)
        return element.value if element else default

    def set_value(self, tag: DicomTag, value: Any):
        """Set value of a data element."""
        self.add_element(tag, value)

    def has_element(self, tag: DicomTag) -> bool:
        """Check if element exists."""
        return tag.tag in self._elements

    def remove_element(self, tag: DicomTag):
        """Remove a data element."""
        if tag.tag in self._elements:
            del self._elements[tag.tag]

    def __getitem__(self, tag: DicomTag) -> Any:
        """Get element value using bracket notation."""
        return self.get_value(tag)

    def __setitem__(self, tag: DicomTag, value: Any):
        """Set element value using bracket notation."""
        self.set_value(tag, value)

    def __contains__(self, tag: DicomTag) -> bool:
        """Check if tag exists using 'in' operator."""
        return self.has_element(tag)

    def __iter__(self) -> Iterator[DicomElement]:
        """Iterate over all elements."""
        return iter(self._elements.values())

    @property
    def pixel_data(self) -> Optional[np.ndarray]:
        """Get pixel data."""
        return self._pixel_data

    @pixel_data.setter
    def pixel_data(self, data: np.ndarray):
        """Set pixel data and update related tags."""
        self._pixel_data = data
        self._update_pixel_tags(data)

    def _update_pixel_tags(self, data: np.ndarray):
        """Update pixel-related tags from array."""
        if data.ndim == 2:
            rows, cols = data.shape
            samples_per_pixel = 1
        elif data.ndim == 3:
            rows, cols, samples_per_pixel = data.shape
        else:
            raise ValueError(f"Invalid pixel data dimensions: {data.ndim}")

        self.set_value(DicomTags.Rows, rows)
        self.set_value(DicomTags.Columns, cols)
        self.set_value(DicomTags.SamplesPerPixel, samples_per_pixel)

        # Determine bits
        if data.dtype == np.uint8:
            bits_allocated = 8
            bits_stored = 8
            high_bit = 7
            pixel_rep = 0
        elif data.dtype == np.uint16:
            bits_allocated = 16
            bits_stored = 16
            high_bit = 15
            pixel_rep = 0
        elif data.dtype == np.int16:
            bits_allocated = 16
            bits_stored = 16
            high_bit = 15
            pixel_rep = 1
        elif data.dtype == np.float32:
            bits_allocated = 32
            bits_stored = 32
            high_bit = 31
            pixel_rep = 0
        else:
            bits_allocated = data.dtype.itemsize * 8
            bits_stored = bits_allocated
            high_bit = bits_stored - 1
            pixel_rep = 0

        self.set_value(DicomTags.BitsAllocated, bits_allocated)
        self.set_value(DicomTags.BitsStored, bits_stored)
        self.set_value(DicomTags.HighBit, high_bit)
        self.set_value(DicomTags.PixelRepresentation, pixel_rep)

        # Photometric interpretation
        if samples_per_pixel == 1:
            self.set_value(DicomTags.PhotometricInterpretation, "MONOCHROME2")
        elif samples_per_pixel == 3:
            self.set_value(DicomTags.PhotometricInterpretation, "RGB")

    def to_dict(self) -> dict[str, Any]:
        """Convert dataset to dictionary."""
        result = {}
        for element in self._elements.values():
            result[element.tag.keyword] = element.value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DicomDataset":
        """Create dataset from dictionary."""
        dataset = cls()
        # Map keywords to tags
        tag_map = {
            attr.keyword: attr
            for attr in vars(DicomTags).values()
            if isinstance(attr, DicomTag)
        }
        for keyword, value in data.items():
            if keyword in tag_map:
                dataset.set_value(tag_map[keyword], value)
        return dataset

    def copy(self) -> "DicomDataset":
        """Create a copy of the dataset."""
        new_dataset = DicomDataset()
        for element in self._elements.values():
            new_dataset.add_element(element.tag, element.value)
        if self._pixel_data is not None:
            new_dataset._pixel_data = self._pixel_data.copy()
        return new_dataset


@dataclass
class DicomSeries:
    """Container for a DICOM series."""

    series_instance_uid: str
    series_number: int = 1
    series_description: str = ""
    modality: str = ""
    body_part: str = ""
    series_date: Optional[date] = None
    series_time: Optional[time] = None
    instances: list[DicomDataset] = field(default_factory=list)

    def __post_init__(self):
        if not self.series_instance_uid:
            self.series_instance_uid = generate_uid("series")

    def add_instance(self, dataset: DicomDataset):
        """Add an instance to the series."""
        # Ensure series-level tags are set
        dataset.set_value(DicomTags.SeriesInstanceUID, self.series_instance_uid)
        dataset.set_value(DicomTags.SeriesNumber, self.series_number)
        dataset.set_value(DicomTags.Modality, self.modality)

        if self.series_description:
            dataset.set_value(DicomTags.SeriesDescription, self.series_description)
        if self.body_part:
            dataset.set_value(DicomTags.BodyPartExamined, self.body_part)

        # Set instance number
        instance_number = len(self.instances) + 1
        dataset.set_value(DicomTags.InstanceNumber, instance_number)

        self.instances.append(dataset)

    def get_instance(self, instance_number: int) -> Optional[DicomDataset]:
        """Get instance by number (1-indexed)."""
        if 1 <= instance_number <= len(self.instances):
            return self.instances[instance_number - 1]
        return None

    @property
    def num_instances(self) -> int:
        """Get number of instances in series."""
        return len(self.instances)


@dataclass
class DicomStudy:
    """Container for a DICOM study."""

    study_instance_uid: str
    study_id: str = ""
    study_description: str = ""
    study_date: Optional[date] = None
    study_time: Optional[time] = None
    accession_number: str = ""
    referring_physician: str = ""
    patient_name: str = ""
    patient_id: str = ""
    patient_birth_date: Optional[date] = None
    patient_sex: str = ""
    series: list[DicomSeries] = field(default_factory=list)

    def __post_init__(self):
        if not self.study_instance_uid:
            self.study_instance_uid = generate_uid("study")

    def add_series(self, series: DicomSeries):
        """Add a series to the study."""
        # Update all instances with study-level info
        for dataset in series.instances:
            self._set_study_tags(dataset)
        self.series.append(series)

    def create_series(
        self,
        modality: str,
        series_description: str = "",
        body_part: str = "",
    ) -> DicomSeries:
        """Create and add a new series."""
        series = DicomSeries(
            series_instance_uid=generate_uid("series"),
            series_number=len(self.series) + 1,
            series_description=series_description,
            modality=modality,
            body_part=body_part,
            series_date=self.study_date,
            series_time=self.study_time,
        )
        self.series.append(series)
        return series

    def _set_study_tags(self, dataset: DicomDataset):
        """Set study-level tags on a dataset."""
        dataset.set_value(DicomTags.StudyInstanceUID, self.study_instance_uid)
        dataset.set_value(DicomTags.StudyID, self.study_id)

        if self.study_description:
            dataset.set_value(DicomTags.StudyDescription, self.study_description)
        if self.study_date:
            dataset.set_value(DicomTags.StudyDate, self.study_date.strftime("%Y%m%d"))
        if self.study_time:
            dataset.set_value(DicomTags.StudyTime, self.study_time.strftime("%H%M%S"))
        if self.accession_number:
            dataset.set_value(DicomTags.AccessionNumber, self.accession_number)
        if self.referring_physician:
            dataset.set_value(DicomTags.ReferringPhysicianName, self.referring_physician)

        # Patient info
        if self.patient_name:
            dataset.set_value(DicomTags.PatientName, self.patient_name)
        if self.patient_id:
            dataset.set_value(DicomTags.PatientID, self.patient_id)
        if self.patient_birth_date:
            dataset.set_value(DicomTags.PatientBirthDate, self.patient_birth_date.strftime("%Y%m%d"))
        if self.patient_sex:
            dataset.set_value(DicomTags.PatientSex, self.patient_sex)

    def get_series(self, series_number: int) -> Optional[DicomSeries]:
        """Get series by number."""
        for s in self.series:
            if s.series_number == series_number:
                return s
        return None

    def get_series_by_modality(self, modality: str) -> list[DicomSeries]:
        """Get all series of a specific modality."""
        return [s for s in self.series if s.modality == modality]

    @property
    def num_series(self) -> int:
        """Get number of series in study."""
        return len(self.series)

    @property
    def total_instances(self) -> int:
        """Get total number of instances across all series."""
        return sum(s.num_instances for s in self.series)
