"""Exam scheduling and resource management."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
from typing import Optional, Callable
import json
import uuid


class ExamPriority(Enum):
    """Exam priority levels."""

    STAT = "stat"
    URGENT = "urgent"
    ROUTINE = "routine"
    LOW = "low"


class ResourceType(Enum):
    """Types of schedulable resources."""

    ROOM = "room"
    DEVICE = "device"
    TECHNOLOGIST = "technologist"


@dataclass
class Resource:
    """Schedulable resource."""

    resource_id: str
    name: str
    resource_type: ResourceType
    modalities: list[str] = field(default_factory=list)
    available_from: time = field(default_factory=lambda: time(8, 0))
    available_to: time = field(default_factory=lambda: time(17, 0))
    working_days: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri
    is_active: bool = True

    def is_available_at(self, dt: datetime) -> bool:
        """Check if resource is available at a given datetime."""
        if not self.is_active:
            return False
        if dt.weekday() not in self.working_days:
            return False
        if not (self.available_from <= dt.time() <= self.available_to):
            return False
        return True


@dataclass
class ScheduledExam:
    """Scheduled examination."""

    exam_id: str
    patient_id: str
    patient_name: str
    modality: str
    exam_description: str
    scheduled_datetime: datetime
    duration_minutes: int = 30
    priority: ExamPriority = ExamPriority.ROUTINE
    room_id: str = ""
    device_id: str = ""
    technologist_id: str = ""
    referring_physician: str = ""
    accession_number: str = ""
    status: str = "scheduled"
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""

    @property
    def end_datetime(self) -> datetime:
        """Get scheduled end time."""
        return self.scheduled_datetime + timedelta(minutes=self.duration_minutes)

    def conflicts_with(self, other: "ScheduledExam") -> bool:
        """Check if this exam conflicts with another."""
        # Check time overlap
        if self.scheduled_datetime >= other.end_datetime:
            return False
        if self.end_datetime <= other.scheduled_datetime:
            return False

        # Check resource conflicts
        if self.room_id and other.room_id and self.room_id == other.room_id:
            return True
        if self.device_id and other.device_id and self.device_id == other.device_id:
            return True
        if self.technologist_id and other.technologist_id and self.technologist_id == other.technologist_id:
            return True

        return False

    def to_dict(self) -> dict:
        return {
            "exam_id": self.exam_id,
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "modality": self.modality,
            "exam_description": self.exam_description,
            "scheduled_datetime": self.scheduled_datetime.isoformat(),
            "duration_minutes": self.duration_minutes,
            "priority": self.priority.value,
            "room_id": self.room_id,
            "device_id": self.device_id,
            "technologist_id": self.technologist_id,
            "referring_physician": self.referring_physician,
            "accession_number": self.accession_number,
            "status": self.status,
            "notes": self.notes,
        }


@dataclass
class TimeSlot:
    """Available time slot."""

    start: datetime
    end: datetime
    room_id: str = ""
    device_id: str = ""

    @property
    def duration_minutes(self) -> int:
        return int((self.end - self.start).total_seconds() / 60)


class Scheduler:
    """Exam scheduling system."""

    def __init__(self, storage_path: str = "schedule.json"):
        self.storage_path = storage_path
        self._exams: dict[str, ScheduledExam] = {}
        self._resources: dict[str, Resource] = {}
        self._listeners: list[Callable[[ScheduledExam, str], None]] = []
        self._load()
        self._create_default_resources()

    def _load(self):
        """Load schedule from storage."""
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
                for e in data.get("exams", []):
                    exam = ScheduledExam(
                        exam_id=e["exam_id"],
                        patient_id=e["patient_id"],
                        patient_name=e["patient_name"],
                        modality=e["modality"],
                        exam_description=e["exam_description"],
                        scheduled_datetime=datetime.fromisoformat(e["scheduled_datetime"]),
                        duration_minutes=e.get("duration_minutes", 30),
                        priority=ExamPriority(e.get("priority", "routine")),
                        room_id=e.get("room_id", ""),
                        device_id=e.get("device_id", ""),
                        technologist_id=e.get("technologist_id", ""),
                        status=e.get("status", "scheduled"),
                    )
                    self._exams[exam.exam_id] = exam
        except FileNotFoundError:
            pass

    def _save(self):
        """Save schedule to storage."""
        data = {
            "exams": [e.to_dict() for e in self._exams.values()]
        }
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _create_default_resources(self):
        """Create default resources."""
        if self._resources:
            return

        # Rooms
        self._resources["ROOM_US1"] = Resource(
            resource_id="ROOM_US1",
            name="Ultrasound Room 1",
            resource_type=ResourceType.ROOM,
            modalities=["US"],
        )
        self._resources["ROOM_XRAY1"] = Resource(
            resource_id="ROOM_XRAY1",
            name="X-Ray Room 1",
            resource_type=ResourceType.ROOM,
            modalities=["DX", "CR"],
        )
        self._resources["ROOM_EYE1"] = Resource(
            resource_id="ROOM_EYE1",
            name="Eye Clinic Room 1",
            resource_type=ResourceType.ROOM,
            modalities=["OPT"],
        )

        # Devices
        self._resources["DEV_US001"] = Resource(
            resource_id="DEV_US001",
            name="Ultrasound System 1",
            resource_type=ResourceType.DEVICE,
            modalities=["US"],
        )
        self._resources["DEV_XRAY001"] = Resource(
            resource_id="DEV_XRAY001",
            name="Portable X-Ray 1",
            resource_type=ResourceType.DEVICE,
            modalities=["DX"],
        )

    def add_listener(self, callback: Callable[[ScheduledExam, str], None]):
        """Add a schedule change listener."""
        self._listeners.append(callback)

    def _notify_listeners(self, exam: ScheduledExam, event: str):
        """Notify listeners of schedule change."""
        for listener in self._listeners:
            try:
                listener(exam, event)
            except Exception:
                pass

    def add_resource(self, resource: Resource):
        """Add a resource to the scheduler."""
        self._resources[resource.resource_id] = resource

    def get_resource(self, resource_id: str) -> Optional[Resource]:
        """Get a resource by ID."""
        return self._resources.get(resource_id)

    def get_resources_by_type(self, resource_type: ResourceType) -> list[Resource]:
        """Get all resources of a type."""
        return [r for r in self._resources.values() if r.resource_type == resource_type and r.is_active]

    def get_resources_by_modality(self, modality: str) -> list[Resource]:
        """Get all resources supporting a modality."""
        return [r for r in self._resources.values() if modality in r.modalities and r.is_active]

    def schedule_exam(
        self,
        patient_id: str,
        patient_name: str,
        modality: str,
        exam_description: str,
        scheduled_datetime: datetime,
        duration_minutes: int = 30,
        priority: ExamPriority = ExamPriority.ROUTINE,
        room_id: str = "",
        device_id: str = "",
        technologist_id: str = "",
        referring_physician: str = "",
        notes: str = "",
        created_by: str = "",
    ) -> ScheduledExam:
        """Schedule a new exam."""
        exam_id = f"EX_{uuid.uuid4().hex[:8].upper()}"
        accession = f"ACC{datetime.now().strftime('%Y%m%d')}{len(self._exams) + 1:04d}"

        exam = ScheduledExam(
            exam_id=exam_id,
            patient_id=patient_id,
            patient_name=patient_name,
            modality=modality,
            exam_description=exam_description,
            scheduled_datetime=scheduled_datetime,
            duration_minutes=duration_minutes,
            priority=priority,
            room_id=room_id,
            device_id=device_id,
            technologist_id=technologist_id,
            referring_physician=referring_physician,
            accession_number=accession,
            notes=notes,
            created_by=created_by,
        )

        # Check for conflicts
        conflicts = self.find_conflicts(exam)
        if conflicts:
            raise ValueError(f"Scheduling conflict with {len(conflicts)} existing exam(s)")

        self._exams[exam_id] = exam
        self._save()
        self._notify_listeners(exam, "scheduled")
        return exam

    def reschedule_exam(
        self,
        exam_id: str,
        new_datetime: datetime,
        new_duration: int = None,
    ) -> bool:
        """Reschedule an existing exam."""
        exam = self._exams.get(exam_id)
        if not exam:
            return False

        old_datetime = exam.scheduled_datetime
        exam.scheduled_datetime = new_datetime
        if new_duration:
            exam.duration_minutes = new_duration

        # Check for conflicts
        conflicts = [e for e in self.find_conflicts(exam) if e.exam_id != exam_id]
        if conflicts:
            exam.scheduled_datetime = old_datetime
            return False

        self._save()
        self._notify_listeners(exam, "rescheduled")
        return True

    def cancel_exam(self, exam_id: str, reason: str = "") -> bool:
        """Cancel a scheduled exam."""
        exam = self._exams.get(exam_id)
        if not exam:
            return False

        exam.status = "cancelled"
        if reason:
            exam.notes += f" Cancelled: {reason}"

        self._save()
        self._notify_listeners(exam, "cancelled")
        return True

    def complete_exam(self, exam_id: str) -> bool:
        """Mark exam as completed."""
        exam = self._exams.get(exam_id)
        if not exam:
            return False

        exam.status = "completed"
        self._save()
        self._notify_listeners(exam, "completed")
        return True

    def get_exam(self, exam_id: str) -> Optional[ScheduledExam]:
        """Get an exam by ID."""
        return self._exams.get(exam_id)

    def get_exam_by_accession(self, accession_number: str) -> Optional[ScheduledExam]:
        """Get an exam by accession number."""
        for exam in self._exams.values():
            if exam.accession_number == accession_number:
                return exam
        return None

    def get_exams_by_date(self, date: datetime) -> list[ScheduledExam]:
        """Get all exams for a specific date."""
        return [
            e for e in self._exams.values()
            if e.scheduled_datetime.date() == date.date() and e.status == "scheduled"
        ]

    def get_exams_by_patient(self, patient_id: str) -> list[ScheduledExam]:
        """Get all exams for a patient."""
        return [e for e in self._exams.values() if e.patient_id == patient_id]

    def get_exams_by_modality(self, modality: str) -> list[ScheduledExam]:
        """Get all exams for a modality."""
        return [e for e in self._exams.values() if e.modality == modality and e.status == "scheduled"]

    def find_conflicts(self, exam: ScheduledExam) -> list[ScheduledExam]:
        """Find scheduling conflicts for an exam."""
        conflicts = []
        for existing in self._exams.values():
            if existing.exam_id == exam.exam_id:
                continue
            if existing.status != "scheduled":
                continue
            if exam.conflicts_with(existing):
                conflicts.append(existing)
        return conflicts

    def find_available_slots(
        self,
        modality: str,
        date: datetime,
        duration_minutes: int = 30,
        room_id: str = "",
    ) -> list[TimeSlot]:
        """Find available time slots for a given date."""
        slots = []

        # Get applicable rooms
        rooms = [r for r in self._resources.values()
                 if r.resource_type == ResourceType.ROOM
                 and modality in r.modalities
                 and r.is_active]

        if room_id:
            rooms = [r for r in rooms if r.resource_id == room_id]

        if not rooms:
            return slots

        # Get existing exams for the date
        day_exams = self.get_exams_by_date(date)

        for room in rooms:
            if not room.is_available_at(date):
                continue

            # Generate slots for this room
            start = datetime.combine(date.date(), room.available_from)
            end = datetime.combine(date.date(), room.available_to)

            current = start
            while current + timedelta(minutes=duration_minutes) <= end:
                slot_end = current + timedelta(minutes=duration_minutes)

                # Check if slot is available
                is_available = True
                for exam in day_exams:
                    if exam.room_id == room.resource_id:
                        if not (slot_end <= exam.scheduled_datetime or current >= exam.end_datetime):
                            is_available = False
                            break

                if is_available:
                    slots.append(TimeSlot(
                        start=current,
                        end=slot_end,
                        room_id=room.resource_id,
                    ))

                current += timedelta(minutes=15)  # 15-minute slot intervals

        return slots

    def get_next_available_slot(
        self,
        modality: str,
        duration_minutes: int = 30,
        priority: ExamPriority = ExamPriority.ROUTINE,
        start_from: datetime = None,
    ) -> Optional[TimeSlot]:
        """Find the next available slot."""
        start = start_from or datetime.now()

        # For STAT/urgent, look at today first
        if priority in (ExamPriority.STAT, ExamPriority.URGENT):
            slots = self.find_available_slots(modality, start, duration_minutes)
            for slot in slots:
                if slot.start >= start:
                    return slot

        # Look at next 7 days
        for day_offset in range(7):
            check_date = start + timedelta(days=day_offset)
            slots = self.find_available_slots(modality, check_date, duration_minutes)
            for slot in slots:
                if slot.start >= start:
                    return slot

        return None

    def get_today_schedule(self, room_id: str = "", modality: str = "") -> list[ScheduledExam]:
        """Get today's schedule."""
        exams = self.get_exams_by_date(datetime.now())

        if room_id:
            exams = [e for e in exams if e.room_id == room_id]
        if modality:
            exams = [e for e in exams if e.modality == modality]

        return sorted(exams, key=lambda e: e.scheduled_datetime)

    def get_statistics(self, date: datetime = None) -> dict:
        """Get scheduling statistics."""
        if date:
            exams = self.get_exams_by_date(date)
        else:
            exams = list(self._exams.values())

        stats = {
            "total": len(exams),
            "by_status": {},
            "by_modality": {},
            "by_priority": {},
            "utilization": {},
        }

        for exam in exams:
            stats["by_status"][exam.status] = stats["by_status"].get(exam.status, 0) + 1
            stats["by_modality"][exam.modality] = stats["by_modality"].get(exam.modality, 0) + 1
            stats["by_priority"][exam.priority.value] = stats["by_priority"].get(exam.priority.value, 0) + 1

        return stats
