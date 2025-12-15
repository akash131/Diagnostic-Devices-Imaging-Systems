"""Acquisition workflow management."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Callable, Any
import uuid


class AcquisitionStatus(Enum):
    """Status of an acquisition step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    REPEATED = "repeated"


@dataclass
class AcquisitionResult:
    """Result of an acquisition step."""

    step_id: str
    status: AcquisitionStatus
    image_ids: list[str] = field(default_factory=list)
    measurements: dict = field(default_factory=dict)
    notes: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    operator: str = ""
    device_id: str = ""
    parameters_used: dict = field(default_factory=dict)
    quality_score: Optional[float] = None

    def duration_seconds(self) -> int:
        """Get duration of acquisition in seconds."""
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at).total_seconds())
        return 0

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "status": self.status.value,
            "image_ids": self.image_ids,
            "measurements": self.measurements,
            "notes": self.notes,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds(),
            "operator": self.operator,
            "device_id": self.device_id,
            "parameters_used": self.parameters_used,
            "quality_score": self.quality_score,
        }


@dataclass
class AcquisitionStep:
    """Step in an acquisition workflow."""

    step_id: str
    name: str
    description: str = ""
    parameters: dict = field(default_factory=dict)
    required: bool = True
    order: int = 0
    status: AcquisitionStatus = AcquisitionStatus.PENDING
    result: Optional[AcquisitionResult] = None
    instructions: str = ""
    repeat_count: int = 0
    max_repeats: int = 3

    def can_repeat(self) -> bool:
        """Check if step can be repeated."""
        return self.repeat_count < self.max_repeats

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "required": self.required,
            "order": self.order,
            "status": self.status.value,
            "result": self.result.to_dict() if self.result else None,
            "instructions": self.instructions,
            "repeat_count": self.repeat_count,
        }


class AcquisitionWorkflow:
    """Manage acquisition workflow for a study."""

    def __init__(
        self,
        study_uid: str,
        protocol_name: str = "",
        operator: str = "",
    ):
        self.workflow_id = str(uuid.uuid4())
        self.study_uid = study_uid
        self.protocol_name = protocol_name
        self.operator = operator
        self.steps: list[AcquisitionStep] = []
        self.current_step_index: int = 0
        self.status = AcquisitionStatus.PENDING
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self._listeners: list[Callable[["AcquisitionWorkflow", AcquisitionStep], None]] = []

    def add_listener(self, callback: Callable[["AcquisitionWorkflow", AcquisitionStep], None]):
        """Add a workflow event listener."""
        self._listeners.append(callback)

    def _notify_listeners(self, step: AcquisitionStep):
        """Notify listeners of step change."""
        for listener in self._listeners:
            try:
                listener(self, step)
            except Exception:
                pass

    def add_step(
        self,
        name: str,
        description: str = "",
        parameters: dict = None,
        required: bool = True,
        instructions: str = "",
    ) -> AcquisitionStep:
        """Add a step to the workflow."""
        step = AcquisitionStep(
            step_id=f"STEP_{len(self.steps) + 1:03d}",
            name=name,
            description=description,
            parameters=parameters or {},
            required=required,
            order=len(self.steps),
            instructions=instructions,
        )
        self.steps.append(step)
        return step

    def get_current_step(self) -> Optional[AcquisitionStep]:
        """Get the current step."""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def get_step_by_id(self, step_id: str) -> Optional[AcquisitionStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def start(self):
        """Start the workflow."""
        self.status = AcquisitionStatus.IN_PROGRESS
        self.started_at = datetime.now()
        if self.steps:
            self.steps[0].status = AcquisitionStatus.IN_PROGRESS

    def start_step(self, step_id: str = None, operator: str = "", device_id: str = "") -> bool:
        """Start an acquisition step."""
        step = self.get_step_by_id(step_id) if step_id else self.get_current_step()
        if not step:
            return False

        step.status = AcquisitionStatus.IN_PROGRESS
        step.result = AcquisitionResult(
            step_id=step.step_id,
            status=AcquisitionStatus.IN_PROGRESS,
            started_at=datetime.now(),
            operator=operator or self.operator,
            device_id=device_id,
            parameters_used=step.parameters.copy(),
        )

        self._notify_listeners(step)
        return True

    def complete_step(
        self,
        step_id: str = None,
        image_ids: list[str] = None,
        measurements: dict = None,
        quality_score: float = None,
        notes: str = "",
    ) -> bool:
        """Complete an acquisition step."""
        step = self.get_step_by_id(step_id) if step_id else self.get_current_step()
        if not step or not step.result:
            return False

        step.status = AcquisitionStatus.COMPLETED
        step.result.status = AcquisitionStatus.COMPLETED
        step.result.completed_at = datetime.now()
        step.result.image_ids = image_ids or []
        step.result.measurements = measurements or {}
        step.result.quality_score = quality_score
        step.result.notes = notes

        # Move to next step
        self._advance_to_next_step()
        self._notify_listeners(step)
        return True

    def skip_step(self, step_id: str = None, reason: str = "") -> bool:
        """Skip a step."""
        step = self.get_step_by_id(step_id) if step_id else self.get_current_step()
        if not step:
            return False

        if step.required:
            return False  # Cannot skip required steps

        step.status = AcquisitionStatus.SKIPPED
        if step.result:
            step.result.status = AcquisitionStatus.SKIPPED
            step.result.notes = reason

        self._advance_to_next_step()
        self._notify_listeners(step)
        return True

    def fail_step(self, step_id: str = None, reason: str = "") -> bool:
        """Mark a step as failed."""
        step = self.get_step_by_id(step_id) if step_id else self.get_current_step()
        if not step:
            return False

        step.status = AcquisitionStatus.FAILED
        if step.result:
            step.result.status = AcquisitionStatus.FAILED
            step.result.completed_at = datetime.now()
            step.result.notes = reason

        self._notify_listeners(step)
        return True

    def repeat_step(self, step_id: str = None, reason: str = "") -> bool:
        """Repeat a step."""
        step = self.get_step_by_id(step_id) if step_id else self.get_current_step()
        if not step or not step.can_repeat():
            return False

        step.repeat_count += 1
        step.status = AcquisitionStatus.PENDING
        step.result = None

        self._notify_listeners(step)
        return True

    def _advance_to_next_step(self):
        """Advance to the next pending step."""
        self.current_step_index += 1

        # Check if workflow is complete
        if self.current_step_index >= len(self.steps):
            self._complete_workflow()
        else:
            # Start next step
            self.steps[self.current_step_index].status = AcquisitionStatus.IN_PROGRESS

    def _complete_workflow(self):
        """Complete the workflow."""
        self.status = AcquisitionStatus.COMPLETED
        self.completed_at = datetime.now()

    def go_to_step(self, step_index: int) -> bool:
        """Go to a specific step."""
        if 0 <= step_index < len(self.steps):
            self.current_step_index = step_index
            return True
        return False

    def is_complete(self) -> bool:
        """Check if workflow is complete."""
        return self.status == AcquisitionStatus.COMPLETED

    def get_progress(self) -> dict:
        """Get workflow progress."""
        completed = sum(1 for s in self.steps if s.status == AcquisitionStatus.COMPLETED)
        skipped = sum(1 for s in self.steps if s.status == AcquisitionStatus.SKIPPED)
        failed = sum(1 for s in self.steps if s.status == AcquisitionStatus.FAILED)

        return {
            "total_steps": len(self.steps),
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
            "pending": len(self.steps) - completed - skipped - failed,
            "current_step": self.current_step_index,
            "progress_percent": (completed + skipped) / len(self.steps) * 100 if self.steps else 0,
        }

    def get_summary(self) -> dict:
        """Get workflow summary."""
        total_duration = 0
        image_count = 0
        all_measurements = {}

        for step in self.steps:
            if step.result:
                total_duration += step.result.duration_seconds()
                image_count += len(step.result.image_ids)
                all_measurements.update(step.result.measurements)

        return {
            "workflow_id": self.workflow_id,
            "study_uid": self.study_uid,
            "protocol_name": self.protocol_name,
            "operator": self.operator,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_duration_seconds": total_duration,
            "image_count": image_count,
            "progress": self.get_progress(),
        }

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "study_uid": self.study_uid,
            "protocol_name": self.protocol_name,
            "operator": self.operator,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "steps": [s.to_dict() for s in self.steps],
            "current_step_index": self.current_step_index,
        }


class WorkflowBuilder:
    """Build acquisition workflows from protocols."""

    @staticmethod
    def from_protocol(
        study_uid: str,
        protocol: Any,  # Protocol from protocol.py
        operator: str = "",
    ) -> AcquisitionWorkflow:
        """Build workflow from a protocol."""
        workflow = AcquisitionWorkflow(
            study_uid=study_uid,
            protocol_name=protocol.name,
            operator=operator,
        )

        for protocol_step in protocol.steps:
            workflow.add_step(
                name=protocol_step.name,
                description=protocol_step.description,
                parameters=protocol_step.parameters.copy(),
                required=protocol_step.required,
                instructions=protocol_step.instructions,
            )

        return workflow

    @staticmethod
    def quick_workflow(
        study_uid: str,
        modality: str,
        views: list[str],
        operator: str = "",
    ) -> AcquisitionWorkflow:
        """Create a quick workflow from a list of views."""
        workflow = AcquisitionWorkflow(
            study_uid=study_uid,
            protocol_name=f"{modality} Quick Acquisition",
            operator=operator,
        )

        for view in views:
            workflow.add_step(
                name=view,
                description=f"Acquire {view} view",
                required=True,
            )

        return workflow
