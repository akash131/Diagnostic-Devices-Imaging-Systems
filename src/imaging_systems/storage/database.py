"""Database layer for medical imaging metadata."""

import sqlite3
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Any
import threading


@dataclass
class PatientRecord:
    """Patient database record."""

    patient_id: str
    patient_name: str
    birth_date: Optional[str] = None
    sex: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


@dataclass
class StudyRecord:
    """Study database record."""

    study_uid: str
    patient_id: str
    study_date: str
    study_time: str = ""
    description: str = ""
    accession_number: str = ""
    referring_physician: str = ""
    modalities: str = ""  # Comma-separated
    num_series: int = 0
    num_instances: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


@dataclass
class SeriesRecord:
    """Series database record."""

    series_uid: str
    study_uid: str
    series_number: int
    modality: str
    description: str = ""
    body_part: str = ""
    num_instances: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


@dataclass
class InstanceRecord:
    """Instance (image) database record."""

    sop_instance_uid: str
    series_uid: str
    instance_number: int
    sop_class_uid: str = ""
    file_path: str = ""
    rows: int = 0
    columns: int = 0
    bits_stored: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


class Database(ABC):
    """Abstract database interface."""

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def create_tables(self):
        pass

    @abstractmethod
    def insert_patient(self, patient: PatientRecord) -> bool:
        pass

    @abstractmethod
    def insert_study(self, study: StudyRecord) -> bool:
        pass

    @abstractmethod
    def insert_series(self, series: SeriesRecord) -> bool:
        pass

    @abstractmethod
    def insert_instance(self, instance: InstanceRecord) -> bool:
        pass

    @abstractmethod
    def get_patient(self, patient_id: str) -> Optional[PatientRecord]:
        pass

    @abstractmethod
    def get_study(self, study_uid: str) -> Optional[StudyRecord]:
        pass

    @abstractmethod
    def get_series(self, series_uid: str) -> Optional[SeriesRecord]:
        pass

    @abstractmethod
    def get_instance(self, sop_instance_uid: str) -> Optional[InstanceRecord]:
        pass

    @abstractmethod
    def search_studies(self, **criteria) -> list[StudyRecord]:
        pass

    @abstractmethod
    def delete_study(self, study_uid: str) -> bool:
        pass


class SQLiteDatabase(Database):
    """SQLite implementation of the database."""

    def __init__(self, db_path: str = "imaging_db.sqlite"):
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    def connect(self) -> bool:
        try:
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self.create_tables()
            return True
        except Exception:
            return False

    def disconnect(self):
        if self._connection:
            self._connection.close()
            self._connection = None

    def create_tables(self):
        if not self._connection:
            return

        cursor = self._connection.cursor()

        # Patients table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                patient_name TEXT NOT NULL,
                birth_date TEXT,
                sex TEXT,
                created_at TEXT,
                metadata TEXT
            )
        """)

        # Studies table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS studies (
                study_uid TEXT PRIMARY KEY,
                patient_id TEXT,
                study_date TEXT,
                study_time TEXT,
                description TEXT,
                accession_number TEXT,
                referring_physician TEXT,
                modalities TEXT,
                num_series INTEGER DEFAULT 0,
                num_instances INTEGER DEFAULT 0,
                created_at TEXT,
                metadata TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
        """)

        # Series table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS series (
                series_uid TEXT PRIMARY KEY,
                study_uid TEXT,
                series_number INTEGER,
                modality TEXT,
                description TEXT,
                body_part TEXT,
                num_instances INTEGER DEFAULT 0,
                created_at TEXT,
                metadata TEXT,
                FOREIGN KEY (study_uid) REFERENCES studies(study_uid)
            )
        """)

        # Instances table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS instances (
                sop_instance_uid TEXT PRIMARY KEY,
                series_uid TEXT,
                instance_number INTEGER,
                sop_class_uid TEXT,
                file_path TEXT,
                rows INTEGER,
                columns INTEGER,
                bits_stored INTEGER,
                created_at TEXT,
                metadata TEXT,
                FOREIGN KEY (series_uid) REFERENCES series(series_uid)
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_studies_patient ON studies(patient_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_studies_date ON studies(study_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_series_study ON series(study_uid)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_instances_series ON instances(series_uid)")

        self._connection.commit()

    def insert_patient(self, patient: PatientRecord) -> bool:
        with self._lock:
            try:
                cursor = self._connection.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO patients
                    (patient_id, patient_name, birth_date, sex, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    patient.patient_id,
                    patient.patient_name,
                    patient.birth_date,
                    patient.sex,
                    patient.created_at.isoformat(),
                    json.dumps(patient.metadata),
                ))
                self._connection.commit()
                return True
            except Exception:
                return False

    def insert_study(self, study: StudyRecord) -> bool:
        with self._lock:
            try:
                cursor = self._connection.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO studies
                    (study_uid, patient_id, study_date, study_time, description,
                     accession_number, referring_physician, modalities, num_series,
                     num_instances, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    study.study_uid,
                    study.patient_id,
                    study.study_date,
                    study.study_time,
                    study.description,
                    study.accession_number,
                    study.referring_physician,
                    study.modalities,
                    study.num_series,
                    study.num_instances,
                    study.created_at.isoformat(),
                    json.dumps(study.metadata),
                ))
                self._connection.commit()
                return True
            except Exception:
                return False

    def insert_series(self, series: SeriesRecord) -> bool:
        with self._lock:
            try:
                cursor = self._connection.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO series
                    (series_uid, study_uid, series_number, modality, description,
                     body_part, num_instances, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    series.series_uid,
                    series.study_uid,
                    series.series_number,
                    series.modality,
                    series.description,
                    series.body_part,
                    series.num_instances,
                    series.created_at.isoformat(),
                    json.dumps(series.metadata),
                ))
                self._connection.commit()
                return True
            except Exception:
                return False

    def insert_instance(self, instance: InstanceRecord) -> bool:
        with self._lock:
            try:
                cursor = self._connection.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO instances
                    (sop_instance_uid, series_uid, instance_number, sop_class_uid,
                     file_path, rows, columns, bits_stored, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    instance.sop_instance_uid,
                    instance.series_uid,
                    instance.instance_number,
                    instance.sop_class_uid,
                    instance.file_path,
                    instance.rows,
                    instance.columns,
                    instance.bits_stored,
                    instance.created_at.isoformat(),
                    json.dumps(instance.metadata),
                ))
                self._connection.commit()
                return True
            except Exception:
                return False

    def get_patient(self, patient_id: str) -> Optional[PatientRecord]:
        cursor = self._connection.cursor()
        cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
        row = cursor.fetchone()
        if row:
            return PatientRecord(
                patient_id=row["patient_id"],
                patient_name=row["patient_name"],
                birth_date=row["birth_date"],
                sex=row["sex"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
        return None

    def get_study(self, study_uid: str) -> Optional[StudyRecord]:
        cursor = self._connection.cursor()
        cursor.execute("SELECT * FROM studies WHERE study_uid = ?", (study_uid,))
        row = cursor.fetchone()
        if row:
            return StudyRecord(
                study_uid=row["study_uid"],
                patient_id=row["patient_id"],
                study_date=row["study_date"],
                study_time=row["study_time"],
                description=row["description"],
                accession_number=row["accession_number"],
                referring_physician=row["referring_physician"],
                modalities=row["modalities"],
                num_series=row["num_series"],
                num_instances=row["num_instances"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
        return None

    def get_series(self, series_uid: str) -> Optional[SeriesRecord]:
        cursor = self._connection.cursor()
        cursor.execute("SELECT * FROM series WHERE series_uid = ?", (series_uid,))
        row = cursor.fetchone()
        if row:
            return SeriesRecord(
                series_uid=row["series_uid"],
                study_uid=row["study_uid"],
                series_number=row["series_number"],
                modality=row["modality"],
                description=row["description"],
                body_part=row["body_part"],
                num_instances=row["num_instances"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
        return None

    def get_instance(self, sop_instance_uid: str) -> Optional[InstanceRecord]:
        cursor = self._connection.cursor()
        cursor.execute("SELECT * FROM instances WHERE sop_instance_uid = ?", (sop_instance_uid,))
        row = cursor.fetchone()
        if row:
            return InstanceRecord(
                sop_instance_uid=row["sop_instance_uid"],
                series_uid=row["series_uid"],
                instance_number=row["instance_number"],
                sop_class_uid=row["sop_class_uid"],
                file_path=row["file_path"],
                rows=row["rows"],
                columns=row["columns"],
                bits_stored=row["bits_stored"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
        return None

    def search_studies(self, **criteria) -> list[StudyRecord]:
        query = "SELECT * FROM studies WHERE 1=1"
        params = []

        if "patient_id" in criteria:
            query += " AND patient_id = ?"
            params.append(criteria["patient_id"])
        if "study_date" in criteria:
            query += " AND study_date = ?"
            params.append(criteria["study_date"])
        if "modality" in criteria:
            query += " AND modalities LIKE ?"
            params.append(f"%{criteria['modality']}%")
        if "date_from" in criteria:
            query += " AND study_date >= ?"
            params.append(criteria["date_from"])
        if "date_to" in criteria:
            query += " AND study_date <= ?"
            params.append(criteria["date_to"])

        query += " ORDER BY study_date DESC"

        cursor = self._connection.cursor()
        cursor.execute(query, params)

        results = []
        for row in cursor.fetchall():
            results.append(StudyRecord(
                study_uid=row["study_uid"],
                patient_id=row["patient_id"],
                study_date=row["study_date"],
                study_time=row["study_time"],
                description=row["description"],
                accession_number=row["accession_number"],
                referring_physician=row["referring_physician"],
                modalities=row["modalities"],
                num_series=row["num_series"],
                num_instances=row["num_instances"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            ))
        return results

    def delete_study(self, study_uid: str) -> bool:
        with self._lock:
            try:
                cursor = self._connection.cursor()
                # Delete instances
                cursor.execute("""
                    DELETE FROM instances WHERE series_uid IN
                    (SELECT series_uid FROM series WHERE study_uid = ?)
                """, (study_uid,))
                # Delete series
                cursor.execute("DELETE FROM series WHERE study_uid = ?", (study_uid,))
                # Delete study
                cursor.execute("DELETE FROM studies WHERE study_uid = ?", (study_uid,))
                self._connection.commit()
                return True
            except Exception:
                return False

    def get_series_for_study(self, study_uid: str) -> list[SeriesRecord]:
        cursor = self._connection.cursor()
        cursor.execute("SELECT * FROM series WHERE study_uid = ? ORDER BY series_number", (study_uid,))

        results = []
        for row in cursor.fetchall():
            results.append(SeriesRecord(
                series_uid=row["series_uid"],
                study_uid=row["study_uid"],
                series_number=row["series_number"],
                modality=row["modality"],
                description=row["description"],
                body_part=row["body_part"],
                num_instances=row["num_instances"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            ))
        return results

    def get_instances_for_series(self, series_uid: str) -> list[InstanceRecord]:
        cursor = self._connection.cursor()
        cursor.execute("SELECT * FROM instances WHERE series_uid = ? ORDER BY instance_number", (series_uid,))

        results = []
        for row in cursor.fetchall():
            results.append(InstanceRecord(
                sop_instance_uid=row["sop_instance_uid"],
                series_uid=row["series_uid"],
                instance_number=row["instance_number"],
                sop_class_uid=row["sop_class_uid"],
                file_path=row["file_path"],
                rows=row["rows"],
                columns=row["columns"],
                bits_stored=row["bits_stored"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            ))
        return results
