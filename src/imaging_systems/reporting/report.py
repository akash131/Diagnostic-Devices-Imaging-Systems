"""Clinical report data structures."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any


class FindingSeverity(Enum):
    """Finding severity levels."""

    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class ReportStatus(Enum):
    """Report status."""

    DRAFT = "draft"
    PRELIMINARY = "preliminary"
    FINAL = "final"
    AMENDED = "amended"
    CANCELLED = "cancelled"


@dataclass
class Finding:
    """Clinical finding in a report."""

    finding_id: str
    description: str
    location: str = ""
    severity: FindingSeverity = FindingSeverity.NORMAL
    code: str = ""  # ICD/SNOMED code
    code_system: str = ""
    measurements: list[dict] = field(default_factory=list)
    images: list[str] = field(default_factory=list)  # Image references
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "finding_id": self.finding_id,
            "description": self.description,
            "location": self.location,
            "severity": self.severity.value,
            "code": self.code,
            "measurements": self.measurements,
        }


@dataclass
class ReportSection:
    """Section of a clinical report."""

    title: str
    content: str = ""
    findings: list[Finding] = field(default_factory=list)
    order: int = 0


@dataclass
class Report:
    """Clinical imaging report."""

    report_id: str
    study_uid: str
    patient_id: str
    patient_name: str

    # Report metadata
    modality: str = ""
    study_description: str = ""
    study_date: str = ""

    # Report content
    title: str = ""
    clinical_history: str = ""
    technique: str = ""
    comparison: str = ""
    sections: list[ReportSection] = field(default_factory=list)
    impression: str = ""
    recommendation: str = ""

    # Status and workflow
    status: ReportStatus = ReportStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    finalized_at: Optional[datetime] = None

    # Personnel
    reporting_physician: str = ""
    referring_physician: str = ""
    technologist: str = ""

    # Additional
    key_images: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_section(self, title: str, content: str = "", order: int = -1):
        """Add a section to the report."""
        if order < 0:
            order = len(self.sections)
        section = ReportSection(title=title, content=content, order=order)
        self.sections.append(section)
        self.sections.sort(key=lambda s: s.order)
        return section

    def add_finding(self, section_title: str, finding: Finding):
        """Add a finding to a section."""
        for section in self.sections:
            if section.title == section_title:
                section.findings.append(finding)
                return
        # Create section if not exists
        section = self.add_section(section_title)
        section.findings.append(finding)

    def finalize(self, physician: str = ""):
        """Finalize the report."""
        self.status = ReportStatus.FINAL
        self.finalized_at = datetime.now()
        if physician:
            self.reporting_physician = physician

    def amend(self, reason: str = ""):
        """Amend a finalized report."""
        if self.status == ReportStatus.FINAL:
            self.status = ReportStatus.AMENDED
            self.metadata["amendment_reason"] = reason
            self.metadata["amendment_date"] = datetime.now().isoformat()

    def get_all_findings(self) -> list[Finding]:
        """Get all findings from all sections."""
        findings = []
        for section in self.sections:
            findings.extend(section.findings)
        return findings

    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "report_id": self.report_id,
            "study_uid": self.study_uid,
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "modality": self.modality,
            "study_date": self.study_date,
            "title": self.title,
            "clinical_history": self.clinical_history,
            "technique": self.technique,
            "sections": [
                {
                    "title": s.title,
                    "content": s.content,
                    "findings": [f.to_dict() for f in s.findings],
                }
                for s in self.sections
            ],
            "impression": self.impression,
            "recommendation": self.recommendation,
            "status": self.status.value,
            "reporting_physician": self.reporting_physician,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ReportTemplate:
    """Template for generating reports."""

    template_id: str
    name: str
    modality: str
    body_part: str = ""
    sections: list[str] = field(default_factory=list)
    default_technique: str = ""
    finding_templates: list[dict] = field(default_factory=list)

    def create_report(
        self,
        report_id: str,
        study_uid: str,
        patient_id: str,
        patient_name: str,
    ) -> Report:
        """Create a new report from this template."""
        report = Report(
            report_id=report_id,
            study_uid=study_uid,
            patient_id=patient_id,
            patient_name=patient_name,
            modality=self.modality,
            technique=self.default_technique,
        )

        for section_title in self.sections:
            report.add_section(section_title)

        return report


# Common templates
ULTRASOUND_TEMPLATE = ReportTemplate(
    template_id="us_general",
    name="General Ultrasound",
    modality="US",
    sections=["Liver", "Gallbladder", "Pancreas", "Spleen", "Kidneys", "Other"],
    default_technique="Gray-scale and color Doppler ultrasound examination was performed.",
)

XRAY_CHEST_TEMPLATE = ReportTemplate(
    template_id="xray_chest",
    name="Chest X-Ray",
    modality="DX",
    body_part="CHEST",
    sections=["Lungs", "Heart", "Mediastinum", "Bones", "Soft Tissues"],
    default_technique="PA and lateral chest radiographs were obtained.",
)

OCT_TEMPLATE = ReportTemplate(
    template_id="oct_macula",
    name="Macular OCT",
    modality="OPT",
    body_part="EYE",
    sections=["Right Eye", "Left Eye", "Comparison"],
    default_technique="Spectral domain OCT imaging of both maculae was performed.",
)

THERMAL_TEMPLATE = ReportTemplate(
    template_id="thermal_general",
    name="Thermal Imaging",
    modality="TG",
    sections=["Temperature Distribution", "Asymmetry Analysis", "Inflammation Assessment"],
    default_technique="Infrared thermal imaging was performed at standardized conditions.",
)
