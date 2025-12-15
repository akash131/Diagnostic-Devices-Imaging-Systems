"""DICOM networking components."""

from dataclasses import dataclass
from typing import Optional, Callable, Any
from enum import Enum
import threading
import socket


class QueryLevel(Enum):
    """DICOM Query/Retrieve levels."""

    PATIENT = "PATIENT"
    STUDY = "STUDY"
    SERIES = "SERIES"
    IMAGE = "IMAGE"


@dataclass
class DicomNode:
    """DICOM network node configuration."""

    ae_title: str
    host: str
    port: int
    description: str = ""


@dataclass
class QueryResult:
    """Result from a DICOM query."""

    level: QueryLevel
    results: list[dict]
    status: str = "Success"
    error: str = ""


class QueryRetrieve:
    """DICOM Query/Retrieve service client."""

    def __init__(self, local_node: DicomNode):
        self.local_node = local_node

    def find(
        self,
        remote_node: DicomNode,
        level: QueryLevel,
        query_filter: dict,
    ) -> QueryResult:
        """Perform C-FIND query."""
        # Simulate query
        results = []

        if level == QueryLevel.STUDY:
            results = [{
                "StudyInstanceUID": "1.2.3.4.5.6",
                "PatientID": query_filter.get("PatientID", "*"),
                "PatientName": query_filter.get("PatientName", "*"),
                "StudyDate": query_filter.get("StudyDate", ""),
                "ModalitiesInStudy": query_filter.get("ModalitiesInStudy", ""),
            }]
        elif level == QueryLevel.SERIES:
            results = [{
                "SeriesInstanceUID": "1.2.3.4.5.6.1",
                "Modality": "US",
                "SeriesNumber": 1,
            }]

        return QueryResult(level=level, results=results)

    def move(
        self,
        remote_node: DicomNode,
        destination_ae: str,
        level: QueryLevel,
        identifiers: dict,
        on_progress: Callable[[int, int], None] = None,
    ) -> dict:
        """Perform C-MOVE retrieval."""
        # Simulate retrieval
        return {
            "success": True,
            "completed": 10,
            "failed": 0,
            "warning": 0,
        }

    def get(
        self,
        remote_node: DicomNode,
        level: QueryLevel,
        identifiers: dict,
        on_instance: Callable[[bytes], None] = None,
    ) -> dict:
        """Perform C-GET retrieval."""
        return {
            "success": True,
            "completed": 5,
        }


class StoreSCU:
    """DICOM Storage Service Class User (sender)."""

    def __init__(self, local_node: DicomNode):
        self.local_node = local_node
        self._socket: Optional[socket.socket] = None

    def send(
        self,
        remote_node: DicomNode,
        dataset_bytes: bytes,
        sop_class_uid: str,
        sop_instance_uid: str,
        on_progress: Callable[[int, int], None] = None,
    ) -> dict:
        """Send a DICOM instance via C-STORE."""
        try:
            # Simulate sending
            return {
                "success": True,
                "sop_instance_uid": sop_instance_uid,
                "status": "Success",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def send_batch(
        self,
        remote_node: DicomNode,
        instances: list[tuple[bytes, str, str]],  # (data, sop_class, sop_instance)
        on_progress: Callable[[int, int], None] = None,
    ) -> dict:
        """Send multiple DICOM instances."""
        results = []
        for i, (data, sop_class, sop_instance) in enumerate(instances):
            result = self.send(remote_node, data, sop_class, sop_instance)
            results.append(result)
            if on_progress:
                on_progress(i + 1, len(instances))

        return {
            "total": len(instances),
            "success": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "results": results,
        }


class StoreSCP:
    """DICOM Storage Service Class Provider (receiver)."""

    def __init__(
        self,
        local_node: DicomNode,
        storage_path: str = "./received",
    ):
        self.local_node = local_node
        self.storage_path = storage_path

        self._running = False
        self._server_socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._on_receive: Optional[Callable[[bytes, dict], None]] = None

    def start(self):
        """Start the SCP server."""
        from pathlib import Path
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)

        self._running = True
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.local_node.host, self.local_node.port))
        self._server_socket.listen(10)

        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the SCP server."""
        self._running = False
        if self._server_socket:
            self._server_socket.close()

    def _listen_loop(self):
        """Main listening loop."""
        while self._running:
            try:
                self._server_socket.settimeout(1.0)
                try:
                    client_socket, address = self._server_socket.accept()
                    threading.Thread(
                        target=self._handle_association,
                        args=(client_socket, address),
                        daemon=True
                    ).start()
                except socket.timeout:
                    continue
            except Exception:
                if self._running:
                    pass

    def _handle_association(self, client_socket: socket.socket, address: tuple):
        """Handle incoming association."""
        try:
            while self._running:
                data = client_socket.recv(65536)
                if not data:
                    break

                # Process received data
                if self._on_receive:
                    self._on_receive(data, {"address": address})

        except Exception:
            pass
        finally:
            client_socket.close()

    def on_receive(self, callback: Callable[[bytes, dict], None]):
        """Register callback for received instances."""
        self._on_receive = callback

    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running


class PrintSCU:
    """DICOM Print Service Class User."""

    def __init__(self, local_node: DicomNode):
        self.local_node = local_node

    def print_image(
        self,
        remote_node: DicomNode,
        image_data: bytes,
        film_size: str = "14INX17IN",
        orientation: str = "PORTRAIT",
        copies: int = 1,
    ) -> dict:
        """Send image to DICOM printer."""
        # Simulate print job
        return {
            "success": True,
            "print_job_id": "PRINT001",
            "status": "Queued",
        }

    def get_printer_status(self, remote_node: DicomNode) -> dict:
        """Query printer status."""
        return {
            "printer_status": "NORMAL",
            "film_consumption": "50%",
            "printer_name": "DICOM Printer",
        }
