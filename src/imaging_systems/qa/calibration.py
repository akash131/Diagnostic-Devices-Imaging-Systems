"""Calibration management and scheduling."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Callable
import json


class CalibrationStatus(Enum):
    """Calibration status."""

    CURRENT = "current"
    DUE_SOON = "due_soon"
    OVERDUE = "overdue"
    NOT_CALIBRATED = "not_calibrated"


@dataclass
class CalibrationRecord:
    """Record of a calibration event."""

    calibration_id: str
    device_id: str
    calibration_type: str
    performed_at: datetime
    performed_by: str
    results: dict = field(default_factory=dict)
    passed: bool = True
    expires_at: Optional[datetime] = None
    notes: str = ""

    def is_expired(self) -> bool:
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False


@dataclass
class CalibrationSchedule:
    """Calibration schedule for a device."""

    device_id: str
    device_name: str
    calibration_type: str
    interval_days: int
    last_calibration: Optional[datetime] = None
    next_due: Optional[datetime] = None
    warning_days: int = 7

    def update_schedule(self, calibration_date: datetime):
        """Update schedule after calibration."""
        self.last_calibration = calibration_date
        self.next_due = calibration_date + timedelta(days=self.interval_days)

    def get_status(self) -> CalibrationStatus:
        """Get current calibration status."""
        if self.last_calibration is None:
            return CalibrationStatus.NOT_CALIBRATED

        if self.next_due is None:
            return CalibrationStatus.NOT_CALIBRATED

        now = datetime.now()
        if now > self.next_due:
            return CalibrationStatus.OVERDUE
        elif now > self.next_due - timedelta(days=self.warning_days):
            return CalibrationStatus.DUE_SOON
        else:
            return CalibrationStatus.CURRENT

    def days_until_due(self) -> int:
        """Get days until calibration is due."""
        if self.next_due is None:
            return 0
        delta = self.next_due - datetime.now()
        return max(0, delta.days)


class CalibrationManager:
    """Manage device calibrations."""

    def __init__(self, storage_path: str = "calibrations.json"):
        self.storage_path = storage_path
        self._schedules: dict[str, CalibrationSchedule] = {}
        self._records: list[CalibrationRecord] = []
        self._load()

    def _load(self):
        """Load calibration data from storage."""
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
                # Reconstruct schedules and records
                for s in data.get("schedules", []):
                    schedule = CalibrationSchedule(
                        device_id=s["device_id"],
                        device_name=s["device_name"],
                        calibration_type=s["calibration_type"],
                        interval_days=s["interval_days"],
                        warning_days=s.get("warning_days", 7),
                    )
                    if s.get("last_calibration"):
                        schedule.last_calibration = datetime.fromisoformat(s["last_calibration"])
                    if s.get("next_due"):
                        schedule.next_due = datetime.fromisoformat(s["next_due"])
                    self._schedules[schedule.device_id] = schedule
        except FileNotFoundError:
            pass

    def _save(self):
        """Save calibration data to storage."""
        data = {
            "schedules": [
                {
                    "device_id": s.device_id,
                    "device_name": s.device_name,
                    "calibration_type": s.calibration_type,
                    "interval_days": s.interval_days,
                    "warning_days": s.warning_days,
                    "last_calibration": s.last_calibration.isoformat() if s.last_calibration else None,
                    "next_due": s.next_due.isoformat() if s.next_due else None,
                }
                for s in self._schedules.values()
            ],
        }
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_device(
        self,
        device_id: str,
        device_name: str,
        calibration_type: str,
        interval_days: int,
    ) -> CalibrationSchedule:
        """Add a device to the calibration schedule."""
        schedule = CalibrationSchedule(
            device_id=device_id,
            device_name=device_name,
            calibration_type=calibration_type,
            interval_days=interval_days,
        )
        self._schedules[device_id] = schedule
        self._save()
        return schedule

    def remove_device(self, device_id: str):
        """Remove a device from calibration tracking."""
        if device_id in self._schedules:
            del self._schedules[device_id]
            self._save()

    def record_calibration(
        self,
        device_id: str,
        performed_by: str,
        results: dict = None,
        passed: bool = True,
        notes: str = "",
    ) -> CalibrationRecord:
        """Record a calibration event."""
        if device_id not in self._schedules:
            raise ValueError(f"Device {device_id} not in calibration schedule")

        schedule = self._schedules[device_id]
        now = datetime.now()

        record = CalibrationRecord(
            calibration_id=f"CAL_{device_id}_{now.strftime('%Y%m%d%H%M%S')}",
            device_id=device_id,
            calibration_type=schedule.calibration_type,
            performed_at=now,
            performed_by=performed_by,
            results=results or {},
            passed=passed,
            expires_at=now + timedelta(days=schedule.interval_days),
            notes=notes,
        )

        self._records.append(record)
        schedule.update_schedule(now)
        self._save()

        return record

    def get_schedule(self, device_id: str) -> Optional[CalibrationSchedule]:
        """Get calibration schedule for a device."""
        return self._schedules.get(device_id)

    def get_all_schedules(self) -> list[CalibrationSchedule]:
        """Get all calibration schedules."""
        return list(self._schedules.values())

    def get_overdue_devices(self) -> list[CalibrationSchedule]:
        """Get list of devices with overdue calibration."""
        return [
            s for s in self._schedules.values()
            if s.get_status() == CalibrationStatus.OVERDUE
        ]

    def get_due_soon_devices(self) -> list[CalibrationSchedule]:
        """Get list of devices with calibration due soon."""
        return [
            s for s in self._schedules.values()
            if s.get_status() == CalibrationStatus.DUE_SOON
        ]

    def get_calibration_history(
        self,
        device_id: str,
        limit: int = 10,
    ) -> list[CalibrationRecord]:
        """Get calibration history for a device."""
        device_records = [r for r in self._records if r.device_id == device_id]
        device_records.sort(key=lambda r: r.performed_at, reverse=True)
        return device_records[:limit]

    def generate_report(self) -> dict:
        """Generate calibration status report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_devices": len(self._schedules),
            "status_summary": {
                "current": 0,
                "due_soon": 0,
                "overdue": 0,
                "not_calibrated": 0,
            },
            "devices": [],
        }

        for schedule in self._schedules.values():
            status = schedule.get_status()
            report["status_summary"][status.value] += 1
            report["devices"].append({
                "device_id": schedule.device_id,
                "device_name": schedule.device_name,
                "calibration_type": schedule.calibration_type,
                "status": status.value,
                "last_calibration": schedule.last_calibration.isoformat() if schedule.last_calibration else None,
                "next_due": schedule.next_due.isoformat() if schedule.next_due else None,
                "days_until_due": schedule.days_until_due(),
            })

        return report


# Default calibration intervals by modality
DEFAULT_INTERVALS = {
    "US": 30,  # Ultrasound: monthly
    "DX": 365,  # X-ray: annual
    "CR": 365,  # Computed Radiography: annual
    "OPT": 90,  # OCT: quarterly
    "TG": 365,  # Thermal: annual
}
