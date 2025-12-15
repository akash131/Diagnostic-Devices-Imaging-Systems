"""DICOM tag definitions and modality-specific tags."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class VR(Enum):
    """DICOM Value Representations."""

    AE = "AE"  # Application Entity
    AS = "AS"  # Age String
    AT = "AT"  # Attribute Tag
    CS = "CS"  # Code String
    DA = "DA"  # Date
    DS = "DS"  # Decimal String
    DT = "DT"  # Date Time
    FL = "FL"  # Floating Point Single
    FD = "FD"  # Floating Point Double
    IS = "IS"  # Integer String
    LO = "LO"  # Long String
    LT = "LT"  # Long Text
    OB = "OB"  # Other Byte
    OD = "OD"  # Other Double
    OF = "OF"  # Other Float
    OW = "OW"  # Other Word
    PN = "PN"  # Person Name
    SH = "SH"  # Short String
    SL = "SL"  # Signed Long
    SQ = "SQ"  # Sequence
    SS = "SS"  # Signed Short
    ST = "ST"  # Short Text
    TM = "TM"  # Time
    UI = "UI"  # Unique Identifier
    UL = "UL"  # Unsigned Long
    UN = "UN"  # Unknown
    US = "US"  # Unsigned Short
    UT = "UT"  # Unlimited Text


@dataclass
class DicomTag:
    """DICOM tag definition."""

    group: int
    element: int
    vr: VR
    name: str
    keyword: str
    vm: str = "1"  # Value Multiplicity

    @property
    def tag(self) -> tuple[int, int]:
        """Return tag as tuple."""
        return (self.group, self.element)

    @property
    def tag_hex(self) -> str:
        """Return tag as hex string."""
        return f"({self.group:04X},{self.element:04X})"

    def __hash__(self):
        return hash((self.group, self.element))


class DicomTags:
    """Standard DICOM tags."""

    # Patient Module
    PatientName = DicomTag(0x0010, 0x0010, VR.PN, "Patient's Name", "PatientName")
    PatientID = DicomTag(0x0010, 0x0020, VR.LO, "Patient ID", "PatientID")
    PatientBirthDate = DicomTag(0x0010, 0x0030, VR.DA, "Patient's Birth Date", "PatientBirthDate")
    PatientSex = DicomTag(0x0010, 0x0040, VR.CS, "Patient's Sex", "PatientSex")
    PatientAge = DicomTag(0x0010, 0x1010, VR.AS, "Patient's Age", "PatientAge")
    PatientWeight = DicomTag(0x0010, 0x1030, VR.DS, "Patient's Weight", "PatientWeight")

    # Study Module
    StudyInstanceUID = DicomTag(0x0020, 0x000D, VR.UI, "Study Instance UID", "StudyInstanceUID")
    StudyDate = DicomTag(0x0008, 0x0020, VR.DA, "Study Date", "StudyDate")
    StudyTime = DicomTag(0x0008, 0x0030, VR.TM, "Study Time", "StudyTime")
    StudyDescription = DicomTag(0x0008, 0x1030, VR.LO, "Study Description", "StudyDescription")
    AccessionNumber = DicomTag(0x0008, 0x0050, VR.SH, "Accession Number", "AccessionNumber")
    ReferringPhysicianName = DicomTag(0x0008, 0x0090, VR.PN, "Referring Physician's Name", "ReferringPhysicianName")
    StudyID = DicomTag(0x0020, 0x0010, VR.SH, "Study ID", "StudyID")

    # Series Module
    SeriesInstanceUID = DicomTag(0x0020, 0x000E, VR.UI, "Series Instance UID", "SeriesInstanceUID")
    SeriesNumber = DicomTag(0x0020, 0x0011, VR.IS, "Series Number", "SeriesNumber")
    SeriesDate = DicomTag(0x0008, 0x0021, VR.DA, "Series Date", "SeriesDate")
    SeriesTime = DicomTag(0x0008, 0x0031, VR.TM, "Series Time", "SeriesTime")
    SeriesDescription = DicomTag(0x0008, 0x103E, VR.LO, "Series Description", "SeriesDescription")
    Modality = DicomTag(0x0008, 0x0060, VR.CS, "Modality", "Modality")
    BodyPartExamined = DicomTag(0x0018, 0x0015, VR.CS, "Body Part Examined", "BodyPartExamined")

    # General Equipment
    Manufacturer = DicomTag(0x0008, 0x0070, VR.LO, "Manufacturer", "Manufacturer")
    InstitutionName = DicomTag(0x0008, 0x0080, VR.LO, "Institution Name", "InstitutionName")
    StationName = DicomTag(0x0008, 0x1010, VR.SH, "Station Name", "StationName")
    ManufacturerModelName = DicomTag(0x0008, 0x1090, VR.LO, "Manufacturer's Model Name", "ManufacturerModelName")
    DeviceSerialNumber = DicomTag(0x0018, 0x1000, VR.LO, "Device Serial Number", "DeviceSerialNumber")
    SoftwareVersions = DicomTag(0x0018, 0x1020, VR.LO, "Software Versions", "SoftwareVersions")

    # Image Module
    SOPClassUID = DicomTag(0x0008, 0x0016, VR.UI, "SOP Class UID", "SOPClassUID")
    SOPInstanceUID = DicomTag(0x0008, 0x0018, VR.UI, "SOP Instance UID", "SOPInstanceUID")
    InstanceNumber = DicomTag(0x0020, 0x0013, VR.IS, "Instance Number", "InstanceNumber")
    ImageType = DicomTag(0x0008, 0x0008, VR.CS, "Image Type", "ImageType", "2-n")
    AcquisitionDate = DicomTag(0x0008, 0x0022, VR.DA, "Acquisition Date", "AcquisitionDate")
    AcquisitionTime = DicomTag(0x0008, 0x0032, VR.TM, "Acquisition Time", "AcquisitionTime")
    ContentDate = DicomTag(0x0008, 0x0023, VR.DA, "Content Date", "ContentDate")
    ContentTime = DicomTag(0x0008, 0x0033, VR.TM, "Content Time", "ContentTime")

    # Image Pixel Module
    Rows = DicomTag(0x0028, 0x0010, VR.US, "Rows", "Rows")
    Columns = DicomTag(0x0028, 0x0011, VR.US, "Columns", "Columns")
    PixelSpacing = DicomTag(0x0028, 0x0030, VR.DS, "Pixel Spacing", "PixelSpacing", "2")
    BitsAllocated = DicomTag(0x0028, 0x0100, VR.US, "Bits Allocated", "BitsAllocated")
    BitsStored = DicomTag(0x0028, 0x0101, VR.US, "Bits Stored", "BitsStored")
    HighBit = DicomTag(0x0028, 0x0102, VR.US, "High Bit", "HighBit")
    PixelRepresentation = DicomTag(0x0028, 0x0103, VR.US, "Pixel Representation", "PixelRepresentation")
    SamplesPerPixel = DicomTag(0x0028, 0x0002, VR.US, "Samples per Pixel", "SamplesPerPixel")
    PhotometricInterpretation = DicomTag(0x0028, 0x0004, VR.CS, "Photometric Interpretation", "PhotometricInterpretation")
    PixelData = DicomTag(0x7FE0, 0x0010, VR.OW, "Pixel Data", "PixelData")

    # Window Level
    WindowCenter = DicomTag(0x0028, 0x1050, VR.DS, "Window Center", "WindowCenter", "1-n")
    WindowWidth = DicomTag(0x0028, 0x1051, VR.DS, "Window Width", "WindowWidth", "1-n")
    RescaleIntercept = DicomTag(0x0028, 0x1052, VR.DS, "Rescale Intercept", "RescaleIntercept")
    RescaleSlope = DicomTag(0x0028, 0x1053, VR.DS, "Rescale Slope", "RescaleSlope")

    # Frame of Reference
    FrameOfReferenceUID = DicomTag(0x0020, 0x0052, VR.UI, "Frame of Reference UID", "FrameOfReferenceUID")
    ImagePositionPatient = DicomTag(0x0020, 0x0032, VR.DS, "Image Position (Patient)", "ImagePositionPatient", "3")
    ImageOrientationPatient = DicomTag(0x0020, 0x0037, VR.DS, "Image Orientation (Patient)", "ImageOrientationPatient", "6")
    SliceThickness = DicomTag(0x0018, 0x0050, VR.DS, "Slice Thickness", "SliceThickness")
    SliceLocation = DicomTag(0x0020, 0x1041, VR.DS, "Slice Location", "SliceLocation")


class ModalityTags:
    """Modality-specific DICOM tags."""

    # Ultrasound Tags
    class US:
        """Ultrasound-specific tags."""

        TransducerType = DicomTag(0x0018, 0x6031, VR.CS, "Transducer Type", "TransducerType")
        TransducerFrequency = DicomTag(0x0018, 0x6030, VR.UL, "Transducer Frequency", "TransducerFrequency")
        DepthOfScanField = DicomTag(0x0018, 0x5050, VR.IS, "Depth of Scan Field", "DepthOfScanField")
        MechanicalIndex = DicomTag(0x0018, 0x5022, VR.DS, "Mechanical Index", "MechanicalIndex")
        ThermalIndex = DicomTag(0x0018, 0x5024, VR.DS, "Thermal Index", "ThermalIndex")
        UltrasoundColorDataPresent = DicomTag(0x0028, 0x0014, VR.US, "Ultrasound Color Data Present", "UltrasoundColorDataPresent")
        NumberOfFrames = DicomTag(0x0028, 0x0008, VR.IS, "Number of Frames", "NumberOfFrames")
        FrameTime = DicomTag(0x0018, 0x1063, VR.DS, "Frame Time", "FrameTime")

    # X-Ray Tags
    class DX:
        """Digital X-ray specific tags."""

        KVP = DicomTag(0x0018, 0x0060, VR.DS, "KVP", "KVP")
        ExposureTime = DicomTag(0x0018, 0x1150, VR.IS, "Exposure Time", "ExposureTime")
        XRayTubeCurrent = DicomTag(0x0018, 0x1151, VR.IS, "X-Ray Tube Current", "XRayTubeCurrent")
        Exposure = DicomTag(0x0018, 0x1152, VR.IS, "Exposure", "Exposure")
        ExposureInuAs = DicomTag(0x0018, 0x1153, VR.IS, "Exposure in µAs", "ExposureInuAs")
        DistanceSourceToDetector = DicomTag(0x0018, 0x1110, VR.DS, "Distance Source to Detector", "DistanceSourceToDetector")
        DistanceSourceToPatient = DicomTag(0x0018, 0x1111, VR.DS, "Distance Source to Patient", "DistanceSourceToPatient")
        Grid = DicomTag(0x0018, 0x1166, VR.CS, "Grid", "Grid")
        FocalSpots = DicomTag(0x0018, 0x1190, VR.DS, "Focal Spot(s)", "FocalSpots")
        DetectorType = DicomTag(0x0018, 0x7004, VR.CS, "Detector Type", "DetectorType")
        DetectorConfiguration = DicomTag(0x0018, 0x7005, VR.CS, "Detector Configuration", "DetectorConfiguration")
        ImagerPixelSpacing = DicomTag(0x0018, 0x1164, VR.DS, "Imager Pixel Spacing", "ImagerPixelSpacing", "2")

    # OCT Tags
    class OPT:
        """Ophthalmic Tomography (OCT) specific tags."""

        AcquisitionDeviceTypeCodeSequence = DicomTag(0x0022, 0x0015, VR.SQ, "Acquisition Device Type Code Sequence", "AcquisitionDeviceTypeCodeSequence")
        LightPathFilterTypeStackCodeSequence = DicomTag(0x0022, 0x0017, VR.SQ, "Light Path Filter Type Stack Code Sequence", "LightPathFilterTypeStackCodeSequence")
        DetectorType = DicomTag(0x0018, 0x7004, VR.CS, "Detector Type", "DetectorType")
        IlluminationWaveLength = DicomTag(0x0022, 0x0055, VR.FL, "Illumination Wave Length", "IlluminationWaveLength")
        IlluminationPower = DicomTag(0x0022, 0x0056, VR.FL, "Illumination Power", "IlluminationPower")
        IlluminationBandwidth = DicomTag(0x0022, 0x0057, VR.FL, "Illumination Bandwidth", "IlluminationBandwidth")
        AxialLengthOfTheEye = DicomTag(0x0022, 0x0030, VR.FL, "Axial Length of the Eye", "AxialLengthOfTheEye")
        DepthSpatialResolution = DicomTag(0x0022, 0x0035, VR.FL, "Depth Spatial Resolution", "DepthSpatialResolution")
        MaximumDepthDistortion = DicomTag(0x0022, 0x0036, VR.FL, "Maximum Depth Distortion", "MaximumDepthDistortion")

    # Thermographic Tags
    class TG:
        """Thermographic imaging tags (using secondary capture with private tags)."""

        # Private creator block
        PrivateCreator = DicomTag(0x0099, 0x0010, VR.LO, "Private Creator", "PrivateCreator")
        SensorType = DicomTag(0x0099, 0x1001, VR.CS, "Sensor Type", "SensorType")
        NETD = DicomTag(0x0099, 0x1002, VR.DS, "NETD", "NETD")
        SpectralRangeMin = DicomTag(0x0099, 0x1003, VR.DS, "Spectral Range Min", "SpectralRangeMin")
        SpectralRangeMax = DicomTag(0x0099, 0x1004, VR.DS, "Spectral Range Max", "SpectralRangeMax")
        Emissivity = DicomTag(0x0099, 0x1005, VR.DS, "Emissivity", "Emissivity")
        AmbientTemperature = DicomTag(0x0099, 0x1006, VR.DS, "Ambient Temperature", "AmbientTemperature")
        MinTemperature = DicomTag(0x0099, 0x1007, VR.DS, "Minimum Temperature", "MinTemperature")
        MaxTemperature = DicomTag(0x0099, 0x1008, VR.DS, "Maximum Temperature", "MaxTemperature")
        DistanceToTarget = DicomTag(0x0099, 0x1009, VR.DS, "Distance to Target", "DistanceToTarget")


class SOPClass:
    """Standard SOP Class UIDs."""

    # Ultrasound
    UltrasoundImageStorage = "1.2.840.10008.5.1.4.1.1.6.1"
    UltrasoundMultiframeImageStorage = "1.2.840.10008.5.1.4.1.1.3.1"

    # X-Ray
    DigitalXRayImageStorageForPresentation = "1.2.840.10008.5.1.4.1.1.1.1"
    DigitalXRayImageStorageForProcessing = "1.2.840.10008.5.1.4.1.1.1.1.1"
    ComputedRadiographyImageStorage = "1.2.840.10008.5.1.4.1.1.1"

    # OCT / Ophthalmic
    OphthalmicTomographyImageStorage = "1.2.840.10008.5.1.4.1.1.77.1.5.4"
    OphthalmicPhotography8BitImageStorage = "1.2.840.10008.5.1.4.1.1.77.1.5.1"

    # Secondary Capture (for thermographic)
    SecondaryCaptureImageStorage = "1.2.840.10008.5.1.4.1.1.7"
    MultiframeSingleBitSecondaryCaptureImageStorage = "1.2.840.10008.5.1.4.1.1.7.1"
    MultiframeGrayscaleByteSecondaryCaptureImageStorage = "1.2.840.10008.5.1.4.1.1.7.2"
    MultiframeGrayscaleWordSecondaryCaptureImageStorage = "1.2.840.10008.5.1.4.1.1.7.3"


class TransferSyntax:
    """DICOM Transfer Syntax UIDs."""

    ImplicitVRLittleEndian = "1.2.840.10008.1.2"
    ExplicitVRLittleEndian = "1.2.840.10008.1.2.1"
    ExplicitVRBigEndian = "1.2.840.10008.1.2.2"
    DeflatedExplicitVRLittleEndian = "1.2.840.10008.1.2.1.99"
    JPEGBaseline = "1.2.840.10008.1.2.4.50"
    JPEGExtended = "1.2.840.10008.1.2.4.51"
    JPEGLossless = "1.2.840.10008.1.2.4.70"
    JPEGLosslessSV1 = "1.2.840.10008.1.2.4.70"
    JPEG2000Lossless = "1.2.840.10008.1.2.4.90"
    JPEG2000 = "1.2.840.10008.1.2.4.91"
    RLELossless = "1.2.840.10008.1.2.5"
