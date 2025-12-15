"""Study management and tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Callable
import json
import uuid


class StudyStatus(Enum):
    """Study workflow status."""

    SCHEDULED = "scheduled"
    CHECKED_IN = "checked_in"
    IN_PROGRESS = "in_progress"
    ACQUISITION_COMPLETE = "acquisition_complete"
    PENDING_REVIEW = "pending_review"
    REVIEWED = "reviewed"
    REPORTED = "reported"
    FINALIZED = "finalized"
    CANCELLED = "cancelled"


@dataclass
class PatientInfo:
    """Patient information for a study."""

    patient_id: str
    patient_name: str
    birth_date: Optional[str] = None
    sex: str = ""
    weight: Optional[float] = None
    height: Optional[float] = None
    allergies: list[str] = field(default_factory=list)
    medical_history: str = ""

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "birth_date": self.birth_date,
            "sex": self.sex,
            "weight": self.weight,
            "height": self.height,
            "allergies": self.allergies,
        }


@dataclass
class Study:
    """Imaging study container."""

    study_instance_uid: str
    patient: PatientInfo
    accession_number: str
    modality: str
    study_description: str = ""
    referring_physician: str = ""
    performing_physician: str = ""
    status: StudyStatus = StudyStatus.SCHEDULED
    scheduled_datetime: Optional[datetime] = None
    started_datetime: Optional[datetime] = None
    completed_datetime: Optional[datetime] = None
    series_uids: list[str] = field(default_factory=list)
    protocol_id: str = ""
    priority: str = "ROUTINE"
    body_part: str = ""
    laterality: str = ""
    reason_for_exam: str = ""
    clinical_history: str = ""
    notes: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_series(self, series_uid: str):
        """Add a series to this study."""
        if series_uid not in self.series_uids:
            self.series_uids.append(series_uid)

    def add_note(self, note: str, author: str = ""):
        """Add a note to the study."""
        timestamp = datetime.now().isoformat()
        self.notes.append(f"[{timestamp}] {author}: {note}" if author else f"[{timestamp}] {note}")

    def start(self):
        """Mark study as started."""
        self.status = StudyStatus.IN_PROGRESS
        self.started_datetime = datetime.now()

    def complete_acquisition(self):
        """Mark acquisition as complete."""
        self.status = StudyStatus.ACQUISITION_COMPLETE
        self.completed_datetime = datetime.now()

    def to_dict(self) -> dict:
        return {
            "study_instance_uid": self.study_instance_uid,
            "patient": self.patient.to_dict(),
            "accession_number": self.accession_number,
            "modality": self.modality,
            "study_description": self.study_description,
            "referring_physician": self.referring_physician,
            "performing_physician": self.performing_physician,
            "status": self.status.value,
            "scheduled_datetime": self.scheduled_datetime.isoformat() if self.scheduled_datetime else None,
            "started_datetime": self.started_datetime.isoformat() if self.started_datetime else None,
            "completed_datetime": self.completed_datetime.isoformat() if self.completed_datetime else None,
            "series_count": len(self.series_uids),
            "protocol_id": self.protocol_id,
            "priority": self.priority,
            "body_part": self.body_part,
        }


class StudyManager:
    """Manage imaging studies."""

    def __init__(self, storage_path: str = "studies.json"):
        self.storage_path = storage_path
        self._studies: dict[str, Study] = {}
        self._listeners: list[Callable[[Study, StudyStatus, StudyStatus], None]] = []
        self._load()

    def _load(self):
        """Load studies from storage."""
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
                for s in data.get("studies", []):
                    patient = PatientInfo(
                        patient_id=s["patient"]["patient_id"],
                        patient_name=s["patient"]["patient_name"],
                        birth_date=s["patient"].get("birth_date"),
                        sex=s["patient"].get("sex", ""),
                    )
                    study = Study(
                        study_instance_uid=s["study_instance_uid"],
                        patient=patient,
                        accession_number=s["accession_number"],
                        modality=s["modality"],
                        study_description=s.get("study_description", ""),
                        status=StudyStatus(s.get("status", "scheduled")),
                        protocol_id=s.get("protocol_id", ""),
                        priority=s.get("priority", "ROUTINE"),
                        body_part=s.get("body_part", ""),
                    )
                    self._studies[study.study_instance_uid] = study
        except FileNotFoundError:
            pass

    def _save(self):
        """Save studies to storage."""
        data = {
            "studies": [s.to_dict() for s in self._studies.values()]
        }
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_listener(self, callback: Callable[[Study, StudyStatus, StudyStatus], None]):
        """Add a status change listener."""
        self._listeners.append(callback)

    def _notify_listeners(self, study: Study, old_status: StudyStatus, new_status: StudyStatus):
        """Notify listeners of status change."""
        for listener in self._listeners:
            try:
                listener(study, old_status, new_status)
            except Exception:
                pass

    def create_study(
        self,
        patient: PatientInfo,
        accession_number: str,
        modality: str,
        study_description: str = "",
        referring_physician: str = "",
        protocol_id: str = "",
        priority: str = "ROUTINE",
        body_part: str = "",
        scheduled_datetime: Optional[datetime] = None,
    ) -> Study:
        """Create a new study."""
        study_uid = f"1.2.840.10008.{uuid.uuid4().int >> 64}"

        study = Study(
            study_instance_uid=study_uid,
            patient=patient,
            accession_number=accession_number,
            modality=modality,
            study_description=study_description,
            referring_physician=referring_physician,
            protocol_id=protocol_id,
            priority=priority,
            body_part=body_part,
            scheduled_datetime=scheduled_datetime,
        )

        self._studies[study_uid] = study
        self._save()
        return study

    def get_study(self, study_uid: str) -> Optional[Study]:
        """Get a study by UID."""
        return self._studies.get(study_uid)

    def get_study_by_accession(self, accession_number: str) -> Optional[Study]:
        """Get a study by accession number."""
        for study in self._studies.values():
            if study.accession_number == accession_number:
                return study
        return None

    def get_studies_by_patient(self, patient_id: str) -> list[Study]:
        """Get all studies for a patient."""
        return [s for s in self._studies.values() if s.patient.patient_id == patient_id]

    def get_studies_by_status(self, status: StudyStatus) -> list[Study]:
        """Get all studies with a specific status."""
        return [s for s in self._studies.values() if s.status == status]

    def get_studies_by_date(self, date: datetime) -> list[Study]:
        """Get studies scheduled for a specific date."""
        return [
            s for s in self._studies.values()
            if s.scheduled_datetime and s.scheduled_datetime.date() == date.date()
        ]

    def update_status(self, study_uid: str, new_status: StudyStatus) -> bool:
        """Update study status."""
        study = self._studies.get(study_uid)
        if not study:
            return False

        old_status = study.status
        study.status = new_status

        # Update timestamps
        if new_status == StudyStatus.IN_PROGRESS and not study.started_datetime:
            study.started_datetime = datetime.now()
        elif new_status in (StudyStatus.ACQUISITION_COMPLETE, StudyStatus.FINALIZED):
            if not study.completed_datetime:
                study.completed_datetime = datetime.now()

        self._save()
        self._notify_listeners(study, old_status, new_status)
        return True

    def check_in_patient(self, study_uid: str) -> bool:
        """Check in a patient for their study."""
        return self.update_status(study_uid, StudyStatus.CHECKED_IN)

    def start_study(self, study_uid: str, performing_physician: str = "") -> bool:
        """Start a study."""
        study = self._studies.get(study_uid)
        if study:
            if performing_physician:
                study.performing_physician = performing_physician
            return self.update_status(study_uid, StudyStatus.IN_PROGRESS)
        return False

    def complete_study(self, study_uid: str) -> bool:
        """Complete study acquisition."""
        return self.update_status(study_uid, StudyStatus.ACQUISITION_COMPLETE)

    def cancel_study(self, study_uid: str, reason: str = "") -> bool:
        """Cancel a study."""
        study = self._studies.get(study_uid)
        if study:
            if reason:
                study.add_note(f"Cancelled: {reason}")
            return self.update_status(study_uid, StudyStatus.CANCELLED)
        return False

    def get_pending_studies(self) -> list[Study]:
        """Get studies pending review."""
        return [
            s for s in self._studies.values()
            if s.status in (StudyStatus.ACQUISITION_COMPLETE, StudyStatus.PENDING_REVIEW)
        ]

    def get_today_schedule(self) -> list[Study]:
        """Get today's scheduled studies."""
        today = datetime.now().date()
        return [
            s for s in self._studies.values()
            if s.scheduled_datetime and s.scheduled_datetime.date() == today
            and s.status in (StudyStatus.SCHEDULED, StudyStatus.CHECKED_IN)
        ]

    def search_studies(
        self,
        patient_name: str = "",
        patient_id: str = "",
        accession_number: str = "",
        modality: str = "",
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> list[Study]:
        """Search for studies matching criteria."""
        results = list(self._studies.values())

        if patient_name:
            results = [s for s in results if patient_name.lower() in s.patient.patient_name.lower()]
        if patient_id:
            results = [s for s in results if patient_id in s.patient.patient_id]
        if accession_number:
            results = [s for s in results if accession_number in s.accession_number]
        if modality:
            results = [s for s in results if s.modality == modality]
        if date_from:
            results = [s for s in results if s.scheduled_datetime and s.scheduled_datetime >= date_from]
        if date_to:
            results = [s for s in results if s.scheduled_datetime and s.scheduled_datetime <= date_to]

        return results

    def get_statistics(self) -> dict:
        """Get study statistics."""
        stats = {
            "total": len(self._studies),
            "by_status": {},
            "by_modality": {},
            "today_scheduled": 0,
            "today_completed": 0,
        }

        today = datetime.now().date()

        for study in self._studies.values():
            # By status
            status = study.status.value
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

            # By modality
            modality = study.modality
            stats["by_modality"][modality] = stats["by_modality"].get(modality, 0) + 1

            # Today
            if study.scheduled_datetime and study.scheduled_datetime.date() == today:
                stats["today_scheduled"] += 1
            if study.completed_datetime and study.completed_datetime.date() == today:
                stats["today_completed"] += 1

        return stats
