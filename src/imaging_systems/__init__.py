"""Imaging Systems Module.

This module provides implementations for various diagnostic imaging devices:
- Ultrasound transducers and signal processing units
- Portable X-ray systems
- Optical coherence tomography (OCT) devices
- Thermographic cameras for inflammation detection

Advanced features include:
- DICOM support for interoperability
- Vendor-specific protocol adapters
- Image processing pipelines
- Database and storage management
- PACS integration and networking
- Clinical reporting
- Quality assurance and calibration
- Workflow management
- CLI tools
- AI/ML integration hooks
"""

# Core device modules
from .base import ImagingDevice, ImageData, DeviceStatus
from .ultrasound import UltrasoundTransducer, SignalProcessor
from .xray import PortableXRay, XRayDetector
from .oct import OCTDevice, OCTScanner
from .thermographic import ThermographicCamera, InflammationDetector

# DICOM support
from .dicom import DicomDataset, DicomBuilder, DicomParser, DicomConverter

# Vendor adapters
from .vendors import VendorAdapter, VendorRegistry, get_vendor_adapter

# Processing pipelines
from .processing import ProcessingPipeline, ImageFilter, ImageAnalyzer, Segmenter

# Storage
from .storage import ImageStore, ImageCache, StudyRecord

# Network/PACS
from .network import PACSClient, PACSServer, DicomNode, WorklistClient

# Reporting
from .reporting import Report, ReportGenerator, StructuredReport

# QA
from .qa import PhantomAnalyzer, ImageQualityAssessment, CalibrationManager

# Workflow
from .workflow import Study, StudyManager, Protocol, ProtocolManager, AcquisitionWorkflow, Scheduler

# CLI
from .cli import CLI, Display, ImageViewer

# AI/ML
from .ai import ModelInference, AIService, Preprocessor

__version__ = "1.0.0"

__all__ = [
    # Core
    "ImagingDevice",
    "ImageData",
    "DeviceStatus",
    # Ultrasound
    "UltrasoundTransducer",
    "SignalProcessor",
    # X-ray
    "PortableXRay",
    "XRayDetector",
    # OCT
    "OCTDevice",
    "OCTScanner",
    # Thermal
    "ThermographicCamera",
    "InflammationDetector",
    # DICOM
    "DicomDataset",
    "DicomBuilder",
    "DicomParser",
    "DicomConverter",
    # Vendors
    "VendorAdapter",
    "VendorRegistry",
    "get_vendor_adapter",
    # Processing
    "ProcessingPipeline",
    "ImageFilter",
    "ImageAnalyzer",
    "Segmenter",
    # Storage
    "ImageStore",
    "ImageCache",
    "StudyRecord",
    # Network
    "PACSClient",
    "PACSServer",
    "DicomNode",
    "WorklistClient",
    # Reporting
    "Report",
    "ReportGenerator",
    "StructuredReport",
    # QA
    "PhantomAnalyzer",
    "ImageQualityAssessment",
    "CalibrationManager",
    # Workflow
    "Study",
    "StudyManager",
    "Protocol",
    "ProtocolManager",
    "AcquisitionWorkflow",
    "Scheduler",
    # CLI
    "CLI",
    "Display",
    "ImageViewer",
    # AI
    "ModelInference",
    "AIService",
    "Preprocessor",
]
