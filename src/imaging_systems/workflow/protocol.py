"""Imaging protocol management."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any
import json
import uuid


class StepType(Enum):
    """Protocol step types."""

    ACQUISITION = "acquisition"
    PROCESSING = "processing"
    MEASUREMENT = "measurement"
    ANNOTATION = "annotation"
    REVIEW = "review"
    EXPORT = "export"


@dataclass
class ProtocolStep:
    """Single step in an imaging protocol."""

    step_id: str
    name: str
    step_type: StepType
    description: str = ""
    parameters: dict = field(default_factory=dict)
    required: bool = True
    order: int = 0
    estimated_duration_seconds: int = 0
    instructions: str = ""

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "step_type": self.step_type.value,
            "description": self.description,
            "parameters": self.parameters,
            "required": self.required,
            "order": self.order,
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "instructions": self.instructions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProtocolStep":
        return cls(
            step_id=data["step_id"],
            name=data["name"],
            step_type=StepType(data["step_type"]),
            description=data.get("description", ""),
            parameters=data.get("parameters", {}),
            required=data.get("required", True),
            order=data.get("order", 0),
            estimated_duration_seconds=data.get("estimated_duration_seconds", 0),
            instructions=data.get("instructions", ""),
        )


@dataclass
class Protocol:
    """Imaging protocol definition."""

    protocol_id: str
    name: str
    modality: str
    body_part: str = ""
    description: str = ""
    version: str = "1.0"
    steps: list[ProtocolStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    is_active: bool = True
    tags: list[str] = field(default_factory=list)
    default_parameters: dict = field(default_factory=dict)

    def add_step(self, step: ProtocolStep):
        """Add a step to the protocol."""
        step.order = len(self.steps)
        self.steps.append(step)
        self.updated_at = datetime.now()

    def remove_step(self, step_id: str):
        """Remove a step from the protocol."""
        self.steps = [s for s in self.steps if s.step_id != step_id]
        # Reorder remaining steps
        for i, step in enumerate(self.steps):
            step.order = i
        self.updated_at = datetime.now()

    def get_step(self, step_id: str) -> Optional[ProtocolStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_acquisition_steps(self) -> list[ProtocolStep]:
        """Get all acquisition steps."""
        return [s for s in self.steps if s.step_type == StepType.ACQUISITION]

    def estimated_duration(self) -> int:
        """Get total estimated duration in seconds."""
        return sum(s.estimated_duration_seconds for s in self.steps)

    def to_dict(self) -> dict:
        return {
            "protocol_id": self.protocol_id,
            "name": self.name,
            "modality": self.modality,
            "body_part": self.body_part,
            "description": self.description,
            "version": self.version,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "is_active": self.is_active,
            "tags": self.tags,
            "default_parameters": self.default_parameters,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Protocol":
        protocol = cls(
            protocol_id=data["protocol_id"],
            name=data["name"],
            modality=data["modality"],
            body_part=data.get("body_part", ""),
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            created_by=data.get("created_by", ""),
            is_active=data.get("is_active", True),
            tags=data.get("tags", []),
            default_parameters=data.get("default_parameters", {}),
        )

        if "created_at" in data:
            protocol.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            protocol.updated_at = datetime.fromisoformat(data["updated_at"])

        for step_data in data.get("steps", []):
            protocol.steps.append(ProtocolStep.from_dict(step_data))

        return protocol


class ProtocolManager:
    """Manage imaging protocols."""

    def __init__(self, storage_path: str = "protocols.json"):
        self.storage_path = storage_path
        self._protocols: dict[str, Protocol] = {}
        self._load()
        self._create_default_protocols()

    def _load(self):
        """Load protocols from storage."""
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
                for p in data.get("protocols", []):
                    protocol = Protocol.from_dict(p)
                    self._protocols[protocol.protocol_id] = protocol
        except FileNotFoundError:
            pass

    def _save(self):
        """Save protocols to storage."""
        data = {
            "protocols": [p.to_dict() for p in self._protocols.values()]
        }
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _create_default_protocols(self):
        """Create default protocols if none exist."""
        if self._protocols:
            return

        # Ultrasound abdomen protocol
        us_abdomen = Protocol(
            protocol_id="US_ABDOMEN_001",
            name="Abdominal Ultrasound",
            modality="US",
            body_part="ABDOMEN",
            description="Standard abdominal ultrasound examination",
        )
        us_abdomen.add_step(ProtocolStep(
            step_id="US_ABD_01",
            name="Liver Survey",
            step_type=StepType.ACQUISITION,
            parameters={"probe": "convex", "frequency": 3.5, "depth": 20},
            estimated_duration_seconds=120,
            instructions="Survey entire liver in multiple planes",
        ))
        us_abdomen.add_step(ProtocolStep(
            step_id="US_ABD_02",
            name="Gallbladder",
            step_type=StepType.ACQUISITION,
            parameters={"probe": "convex", "frequency": 5.0, "depth": 15},
            estimated_duration_seconds=60,
            instructions="Image gallbladder longitudinal and transverse",
        ))
        us_abdomen.add_step(ProtocolStep(
            step_id="US_ABD_03",
            name="Kidneys",
            step_type=StepType.ACQUISITION,
            parameters={"probe": "convex", "frequency": 3.5, "depth": 18},
            estimated_duration_seconds=120,
            instructions="Image both kidneys in long and short axis",
        ))
        self._protocols[us_abdomen.protocol_id] = us_abdomen

        # Chest X-ray protocol
        dx_chest = Protocol(
            protocol_id="DX_CHEST_001",
            name="Chest X-Ray PA/Lateral",
            modality="DX",
            body_part="CHEST",
            description="Standard chest radiograph",
        )
        dx_chest.add_step(ProtocolStep(
            step_id="DX_CHEST_PA",
            name="PA View",
            step_type=StepType.ACQUISITION,
            parameters={"kvp": 120, "mas": 2.5, "distance": 180},
            estimated_duration_seconds=30,
            instructions="Patient standing, arms at sides, full inspiration",
        ))
        dx_chest.add_step(ProtocolStep(
            step_id="DX_CHEST_LAT",
            name="Lateral View",
            step_type=StepType.ACQUISITION,
            parameters={"kvp": 125, "mas": 5.0, "distance": 180},
            estimated_duration_seconds=30,
            instructions="Left lateral, arms raised above head",
        ))
        self._protocols[dx_chest.protocol_id] = dx_chest

        # OCT Macula protocol
        oct_macula = Protocol(
            protocol_id="OCT_MACULA_001",
            name="Macular OCT",
            modality="OPT",
            body_part="EYE",
            description="Standard macular OCT scan",
        )
        oct_macula.add_step(ProtocolStep(
            step_id="OCT_MAC_01",
            name="Macular Cube",
            step_type=StepType.ACQUISITION,
            parameters={"scan_pattern": "cube", "size_mm": 6.0, "b_scans": 128},
            estimated_duration_seconds=10,
            instructions="Center on fovea, optimize signal strength",
        ))
        oct_macula.add_step(ProtocolStep(
            step_id="OCT_MAC_02",
            name="Radial Lines",
            step_type=StepType.ACQUISITION,
            parameters={"scan_pattern": "radial", "lines": 12, "length_mm": 6.0},
            estimated_duration_seconds=15,
            instructions="Radial pattern through fovea",
        ))
        self._protocols[oct_macula.protocol_id] = oct_macula

        # Thermal inflammation protocol
        tg_inflam = Protocol(
            protocol_id="TG_INFLAM_001",
            name="Inflammation Assessment",
            modality="TG",
            body_part="",
            description="Thermal imaging for inflammation detection",
        )
        tg_inflam.add_step(ProtocolStep(
            step_id="TG_INF_01",
            name="Baseline Capture",
            step_type=StepType.ACQUISITION,
            parameters={"integration_time": 50, "resolution": "high"},
            estimated_duration_seconds=5,
            instructions="Capture baseline thermal image of region",
        ))
        tg_inflam.add_step(ProtocolStep(
            step_id="TG_INF_02",
            name="Analysis",
            step_type=StepType.PROCESSING,
            parameters={"threshold": 1.5, "min_area": 100},
            estimated_duration_seconds=10,
            instructions="Run inflammation detection algorithm",
        ))
        self._protocols[tg_inflam.protocol_id] = tg_inflam

        self._save()

    def create_protocol(
        self,
        name: str,
        modality: str,
        body_part: str = "",
        description: str = "",
        created_by: str = "",
    ) -> Protocol:
        """Create a new protocol."""
        protocol_id = f"{modality}_{name[:10].upper().replace(' ', '_')}_{uuid.uuid4().hex[:6]}"

        protocol = Protocol(
            protocol_id=protocol_id,
            name=name,
            modality=modality,
            body_part=body_part,
            description=description,
            created_by=created_by,
        )

        self._protocols[protocol_id] = protocol
        self._save()
        return protocol

    def get_protocol(self, protocol_id: str) -> Optional[Protocol]:
        """Get a protocol by ID."""
        return self._protocols.get(protocol_id)

    def get_protocols_by_modality(self, modality: str) -> list[Protocol]:
        """Get all protocols for a modality."""
        return [p for p in self._protocols.values() if p.modality == modality and p.is_active]

    def get_protocols_by_body_part(self, body_part: str) -> list[Protocol]:
        """Get all protocols for a body part."""
        return [p for p in self._protocols.values() if p.body_part == body_part and p.is_active]

    def search_protocols(
        self,
        name: str = "",
        modality: str = "",
        body_part: str = "",
        tags: list[str] = None,
    ) -> list[Protocol]:
        """Search for protocols."""
        results = [p for p in self._protocols.values() if p.is_active]

        if name:
            results = [p for p in results if name.lower() in p.name.lower()]
        if modality:
            results = [p for p in results if p.modality == modality]
        if body_part:
            results = [p for p in results if p.body_part == body_part]
        if tags:
            results = [p for p in results if any(t in p.tags for t in tags)]

        return results

    def update_protocol(self, protocol: Protocol):
        """Update a protocol."""
        protocol.updated_at = datetime.now()
        self._protocols[protocol.protocol_id] = protocol
        self._save()

    def delete_protocol(self, protocol_id: str) -> bool:
        """Delete a protocol (soft delete)."""
        protocol = self._protocols.get(protocol_id)
        if protocol:
            protocol.is_active = False
            self._save()
            return True
        return False

    def duplicate_protocol(self, protocol_id: str, new_name: str) -> Optional[Protocol]:
        """Duplicate a protocol with a new name."""
        original = self._protocols.get(protocol_id)
        if not original:
            return None

        new_protocol = Protocol(
            protocol_id=f"{original.modality}_{new_name[:10].upper().replace(' ', '_')}_{uuid.uuid4().hex[:6]}",
            name=new_name,
            modality=original.modality,
            body_part=original.body_part,
            description=original.description,
            tags=original.tags.copy(),
            default_parameters=original.default_parameters.copy(),
        )

        for step in original.steps:
            new_step = ProtocolStep(
                step_id=f"{step.step_id}_copy",
                name=step.name,
                step_type=step.step_type,
                description=step.description,
                parameters=step.parameters.copy(),
                required=step.required,
                order=step.order,
                estimated_duration_seconds=step.estimated_duration_seconds,
                instructions=step.instructions,
            )
            new_protocol.steps.append(new_step)

        self._protocols[new_protocol.protocol_id] = new_protocol
        self._save()
        return new_protocol

    def export_protocol(self, protocol_id: str) -> Optional[str]:
        """Export a protocol to JSON string."""
        protocol = self._protocols.get(protocol_id)
        if protocol:
            return json.dumps(protocol.to_dict(), indent=2)
        return None

    def import_protocol(self, json_str: str) -> Protocol:
        """Import a protocol from JSON string."""
        data = json.loads(json_str)
        # Generate new ID to avoid conflicts
        data["protocol_id"] = f"{data['modality']}_{uuid.uuid4().hex[:8]}"

        protocol = Protocol.from_dict(data)
        self._protocols[protocol.protocol_id] = protocol
        self._save()
        return protocol

    def get_all_protocols(self, include_inactive: bool = False) -> list[Protocol]:
        """Get all protocols."""
        if include_inactive:
            return list(self._protocols.values())
        return [p for p in self._protocols.values() if p.is_active]
