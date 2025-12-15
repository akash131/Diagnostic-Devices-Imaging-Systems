"""DICOM UID generation utilities."""

import hashlib
import os
import time
import uuid
from datetime import datetime
from typing import Optional


class UIDGenerator:
    """Generator for DICOM UIDs."""

    # Organization root - should be registered with DICOM
    # Using example root for demonstration
    ORG_ROOT = "1.2.826.0.1.3680043.9.7433"

    def __init__(self, org_root: Optional[str] = None):
        """Initialize UID generator with organization root."""
        self.org_root = org_root or self.ORG_ROOT
        self._counter = 0

    def generate_study_uid(self) -> str:
        """Generate a Study Instance UID."""
        return self._generate_uid("1")

    def generate_series_uid(self) -> str:
        """Generate a Series Instance UID."""
        return self._generate_uid("2")

    def generate_sop_uid(self) -> str:
        """Generate a SOP Instance UID."""
        return self._generate_uid("3")

    def generate_frame_of_reference_uid(self) -> str:
        """Generate a Frame of Reference UID."""
        return self._generate_uid("4")

    def _generate_uid(self, type_code: str) -> str:
        """Generate a unique UID."""
        self._counter += 1

        # Components for uniqueness
        timestamp = int(time.time() * 1000000)
        process_id = os.getpid()
        random_part = uuid.uuid4().int % 1000000000

        # Build UID
        uid = f"{self.org_root}.{type_code}.{timestamp}.{process_id}.{random_part}.{self._counter}"

        # Ensure UID doesn't exceed 64 characters
        if len(uid) > 64:
            # Use hash-based approach if too long
            hash_input = f"{timestamp}{process_id}{random_part}{self._counter}"
            hash_value = hashlib.md5(hash_input.encode()).hexdigest()
            # Convert hex to decimal digits
            decimal_hash = str(int(hash_value[:16], 16))
            uid = f"{self.org_root}.{type_code}.{decimal_hash}"

        return uid[:64]  # Truncate to 64 chars max

    def generate_uid_from_data(self, data: bytes) -> str:
        """Generate reproducible UID from data."""
        hash_value = hashlib.sha256(data).hexdigest()
        decimal_hash = str(int(hash_value[:20], 16))
        uid = f"{self.org_root}.5.{decimal_hash}"
        return uid[:64]


# Global generator instance
_default_generator = UIDGenerator()


def generate_uid(uid_type: str = "sop") -> str:
    """Generate a DICOM UID.

    Args:
        uid_type: Type of UID ('study', 'series', 'sop', 'frame_of_reference')

    Returns:
        Generated UID string
    """
    if uid_type == "study":
        return _default_generator.generate_study_uid()
    elif uid_type == "series":
        return _default_generator.generate_series_uid()
    elif uid_type == "sop":
        return _default_generator.generate_sop_uid()
    elif uid_type == "frame_of_reference":
        return _default_generator.generate_frame_of_reference_uid()
    else:
        return _default_generator._generate_uid("9")


def set_organization_root(org_root: str):
    """Set the organization root for UID generation."""
    global _default_generator
    _default_generator = UIDGenerator(org_root)


class UIDRegistry:
    """Registry to track generated UIDs for consistency."""

    def __init__(self):
        self._study_uids: dict[str, str] = {}
        self._series_uids: dict[str, str] = {}
        self._instance_uids: dict[str, str] = {}

    def get_or_create_study_uid(self, study_key: str) -> str:
        """Get existing or create new Study UID."""
        if study_key not in self._study_uids:
            self._study_uids[study_key] = generate_uid("study")
        return self._study_uids[study_key]

    def get_or_create_series_uid(self, series_key: str) -> str:
        """Get existing or create new Series UID."""
        if series_key not in self._series_uids:
            self._series_uids[series_key] = generate_uid("series")
        return self._series_uids[series_key]

    def get_or_create_instance_uid(self, instance_key: str) -> str:
        """Get existing or create new SOP Instance UID."""
        if instance_key not in self._instance_uids:
            self._instance_uids[instance_key] = generate_uid("sop")
        return self._instance_uids[instance_key]

    def clear(self):
        """Clear all cached UIDs."""
        self._study_uids.clear()
        self._series_uids.clear()
        self._instance_uids.clear()
