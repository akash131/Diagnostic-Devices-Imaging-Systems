"""DICOM Structured Report support."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any


class MeasurementUnit(Enum):
    """Common measurement units."""

    MM = ("mm", "millimeter")
    CM = ("cm", "centimeter")
    M = ("m", "meter")
    MM2 = ("mm2", "square millimeter")
    CM2 = ("cm2", "square centimeter")
    MM3 = ("mm3", "cubic millimeter")
    ML = ("ml", "milliliter")
    DEGREE = ("deg", "degree")
    PERCENT = ("%", "percent")
    CELSIUS = ("Cel", "degree Celsius")
    HU = ("HU", "Hounsfield unit")
    DB = ("dB", "decibel")


@dataclass
class CodedConcept:
    """DICOM coded concept (code value, scheme, meaning)."""

    value: str
    scheme_designator: str
    meaning: str

    def to_dict(self) -> dict:
        return {
            "CodeValue": self.value,
            "CodingSchemeDesignator": self.scheme_designator,
            "CodeMeaning": self.meaning,
        }


@dataclass
class Measurement:
    """A measurement value with units."""

    name: str
    value: float
    unit: MeasurementUnit
    concept_code: Optional[CodedConcept] = None
    reference_range: Optional[tuple[float, float]] = None
    location: str = ""
    method: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def is_normal(self) -> Optional[bool]:
        """Check if value is within reference range."""
        if self.reference_range:
            return self.reference_range[0] <= self.value <= self.reference_range[1]
        return None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit.value[0],
            "unit_name": self.unit.value[1],
            "location": self.location,
            "is_normal": self.is_normal(),
        }


@dataclass
class SRContentItem:
    """Structured Report content item."""

    value_type: str  # TEXT, NUM, CODE, IMAGE, etc.
    concept_name: CodedConcept
    value: Any
    relationship_type: str = "CONTAINS"  # CONTAINS, HAS OBS CONTEXT, etc.
    children: list["SRContentItem"] = field(default_factory=list)

    def add_child(self, item: "SRContentItem"):
        self.children.append(item)


class SRTemplate:
    """DICOM SR Template."""

    def __init__(self, template_id: str, name: str):
        self.template_id = template_id
        self.name = name
        self._root_items: list[SRContentItem] = []

    def add_text(
        self,
        concept_name: CodedConcept,
        text: str,
        parent: Optional[SRContentItem] = None,
    ) -> SRContentItem:
        """Add a text content item."""
        item = SRContentItem(
            value_type="TEXT",
            concept_name=concept_name,
            value=text,
        )
        if parent:
            parent.add_child(item)
        else:
            self._root_items.append(item)
        return item

    def add_numeric(
        self,
        concept_name: CodedConcept,
        value: float,
        unit: MeasurementUnit,
        parent: Optional[SRContentItem] = None,
    ) -> SRContentItem:
        """Add a numeric content item."""
        item = SRContentItem(
            value_type="NUM",
            concept_name=concept_name,
            value={"value": value, "unit": unit.value[0]},
        )
        if parent:
            parent.add_child(item)
        else:
            self._root_items.append(item)
        return item

    def add_code(
        self,
        concept_name: CodedConcept,
        value: CodedConcept,
        parent: Optional[SRContentItem] = None,
    ) -> SRContentItem:
        """Add a coded content item."""
        item = SRContentItem(
            value_type="CODE",
            concept_name=concept_name,
            value=value,
        )
        if parent:
            parent.add_child(item)
        else:
            self._root_items.append(item)
        return item

    def add_container(
        self,
        concept_name: CodedConcept,
        parent: Optional[SRContentItem] = None,
    ) -> SRContentItem:
        """Add a container content item."""
        item = SRContentItem(
            value_type="CONTAINER",
            concept_name=concept_name,
            value=None,
        )
        if parent:
            parent.add_child(item)
        else:
            self._root_items.append(item)
        return item


class StructuredReport:
    """DICOM Structured Report document."""

    def __init__(
        self,
        report_id: str,
        template: Optional[SRTemplate] = None,
    ):
        self.report_id = report_id
        self.template = template

        # Document metadata
        self.sop_class_uid = "1.2.840.10008.5.1.4.1.1.88.33"  # Basic Text SR
        self.sop_instance_uid = ""
        self.study_uid = ""
        self.series_uid = ""
        self.patient_id = ""
        self.patient_name = ""

        # Content
        self.content_items: list[SRContentItem] = []
        self.measurements: list[Measurement] = []

        # Timestamps
        self.content_date = datetime.now().strftime("%Y%m%d")
        self.content_time = datetime.now().strftime("%H%M%S")

    def add_finding(
        self,
        finding_text: str,
        concept_code: Optional[CodedConcept] = None,
    ) -> SRContentItem:
        """Add a finding to the report."""
        concept = concept_code or CodedConcept(
            "121071", "DCM", "Finding"
        )
        item = SRContentItem(
            value_type="TEXT",
            concept_name=concept,
            value=finding_text,
        )
        self.content_items.append(item)
        return item

    def add_measurement(self, measurement: Measurement) -> SRContentItem:
        """Add a measurement to the report."""
        self.measurements.append(measurement)

        concept = measurement.concept_code or CodedConcept(
            "121206", "DCM", "Distance"
        )
        item = SRContentItem(
            value_type="NUM",
            concept_name=concept,
            value={
                "value": measurement.value,
                "unit": measurement.unit.value[0],
            },
        )
        self.content_items.append(item)
        return item

    def add_impression(self, impression_text: str) -> SRContentItem:
        """Add impression/conclusion."""
        item = SRContentItem(
            value_type="TEXT",
            concept_name=CodedConcept("121073", "DCM", "Impression"),
            value=impression_text,
        )
        self.content_items.append(item)
        return item

    def add_image_reference(
        self,
        sop_instance_uid: str,
        sop_class_uid: str,
        frame_number: Optional[int] = None,
    ) -> SRContentItem:
        """Add reference to an image."""
        item = SRContentItem(
            value_type="IMAGE",
            concept_name=CodedConcept("121191", "DCM", "Referenced Image"),
            value={
                "ReferencedSOPClassUID": sop_class_uid,
                "ReferencedSOPInstanceUID": sop_instance_uid,
                "ReferencedFrameNumber": frame_number,
            },
        )
        self.content_items.append(item)
        return item

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "report_id": self.report_id,
            "sop_class_uid": self.sop_class_uid,
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "study_uid": self.study_uid,
            "content_date": self.content_date,
            "content_time": self.content_time,
            "measurements": [m.to_dict() for m in self.measurements],
            "content_items": len(self.content_items),
        }


# Common coded concepts
class SRConcepts:
    """Common SR coded concepts."""

    # Document titles
    BASIC_DIAGNOSTIC_IMAGING_REPORT = CodedConcept(
        "11528-7", "LN", "Radiology Report"
    )

    # Finding-related
    FINDING = CodedConcept("121071", "DCM", "Finding")
    IMPRESSION = CodedConcept("121073", "DCM", "Impression")
    RECOMMENDATION = CodedConcept("121074", "DCM", "Recommendation")

    # Measurements
    DISTANCE = CodedConcept("121206", "DCM", "Distance")
    AREA = CodedConcept("121207", "DCM", "Area")
    VOLUME = CodedConcept("121216", "DCM", "Volume")
    DIAMETER = CodedConcept("121211", "DCM", "Diameter")
    LENGTH = CodedConcept("121207", "DCM", "Length")

    # Anatomical locations
    LIVER = CodedConcept("10200004", "SCT", "Liver")
    KIDNEY = CodedConcept("64033007", "SCT", "Kidney")
    HEART = CodedConcept("80891009", "SCT", "Heart")
    LUNG = CodedConcept("39607008", "SCT", "Lung")

    # Image references
    REFERENCED_IMAGE = CodedConcept("121191", "DCM", "Referenced Image")
    KEY_IMAGE = CodedConcept("121180", "DCM", "Key Object Selection")
