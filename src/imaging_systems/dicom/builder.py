"""DICOM dataset builder for creating compliant DICOM objects."""

from datetime import datetime, date, time
from typing import Optional, Any
import numpy as np

from .dataset import DicomDataset, DicomSeries, DicomStudy
from .tags import DicomTags, ModalityTags, SOPClass, TransferSyntax
from .uid import generate_uid


class DicomBuilder:
    """Builder for creating DICOM datasets."""

    def __init__(self):
        self._dataset = DicomDataset()
        self._set_defaults()

    def _set_defaults(self):
        """Set default required elements."""
        now = datetime.now()
        self._dataset.set_value(DicomTags.SOPInstanceUID, generate_uid("sop"))
        self._dataset.set_value(DicomTags.StudyDate, now.strftime("%Y%m%d"))
        self._dataset.set_value(DicomTags.StudyTime, now.strftime("%H%M%S"))
        self._dataset.set_value(DicomTags.SeriesDate, now.strftime("%Y%m%d"))
        self._dataset.set_value(DicomTags.SeriesTime, now.strftime("%H%M%S"))
        self._dataset.set_value(DicomTags.ContentDate, now.strftime("%Y%m%d"))
        self._dataset.set_value(DicomTags.ContentTime, now.strftime("%H%M%S"))

    def set_patient(
        self,
        name: str,
        patient_id: str,
        birth_date: Optional[date] = None,
        sex: str = "",
        age: str = "",
        weight: Optional[float] = None,
    ) -> "DicomBuilder":
        """Set patient information."""
        self._dataset.set_value(DicomTags.PatientName, name)
        self._dataset.set_value(DicomTags.PatientID, patient_id)
        if birth_date:
            self._dataset.set_value(DicomTags.PatientBirthDate, birth_date.strftime("%Y%m%d"))
        if sex:
            self._dataset.set_value(DicomTags.PatientSex, sex)
        if age:
            self._dataset.set_value(DicomTags.PatientAge, age)
        if weight:
            self._dataset.set_value(DicomTags.PatientWeight, str(weight))
        return self

    def set_study(
        self,
        study_uid: Optional[str] = None,
        study_id: str = "",
        description: str = "",
        accession_number: str = "",
        referring_physician: str = "",
        study_date: Optional[date] = None,
        study_time: Optional[time] = None,
    ) -> "DicomBuilder":
        """Set study information."""
        self._dataset.set_value(DicomTags.StudyInstanceUID, study_uid or generate_uid("study"))
        if study_id:
            self._dataset.set_value(DicomTags.StudyID, study_id)
        if description:
            self._dataset.set_value(DicomTags.StudyDescription, description)
        if accession_number:
            self._dataset.set_value(DicomTags.AccessionNumber, accession_number)
        if referring_physician:
            self._dataset.set_value(DicomTags.ReferringPhysicianName, referring_physician)
        if study_date:
            self._dataset.set_value(DicomTags.StudyDate, study_date.strftime("%Y%m%d"))
        if study_time:
            self._dataset.set_value(DicomTags.StudyTime, study_time.strftime("%H%M%S"))
        return self

    def set_series(
        self,
        series_uid: Optional[str] = None,
        series_number: int = 1,
        modality: str = "",
        description: str = "",
        body_part: str = "",
    ) -> "DicomBuilder":
        """Set series information."""
        self._dataset.set_value(DicomTags.SeriesInstanceUID, series_uid or generate_uid("series"))
        self._dataset.set_value(DicomTags.SeriesNumber, series_number)
        if modality:
            self._dataset.set_value(DicomTags.Modality, modality)
        if description:
            self._dataset.set_value(DicomTags.SeriesDescription, description)
        if body_part:
            self._dataset.set_value(DicomTags.BodyPartExamined, body_part)
        return self

    def set_equipment(
        self,
        manufacturer: str = "",
        institution: str = "",
        station_name: str = "",
        model_name: str = "",
        serial_number: str = "",
        software_version: str = "",
    ) -> "DicomBuilder":
        """Set equipment information."""
        if manufacturer:
            self._dataset.set_value(DicomTags.Manufacturer, manufacturer)
        if institution:
            self._dataset.set_value(DicomTags.InstitutionName, institution)
        if station_name:
            self._dataset.set_value(DicomTags.StationName, station_name)
        if model_name:
            self._dataset.set_value(DicomTags.ManufacturerModelName, model_name)
        if serial_number:
            self._dataset.set_value(DicomTags.DeviceSerialNumber, serial_number)
        if software_version:
            self._dataset.set_value(DicomTags.SoftwareVersions, software_version)
        return self

    def set_image(
        self,
        pixel_data: np.ndarray,
        pixel_spacing: Optional[tuple[float, float]] = None,
        slice_thickness: Optional[float] = None,
        image_position: Optional[tuple[float, float, float]] = None,
        image_orientation: Optional[tuple[float, ...]] = None,
        window_center: Optional[float] = None,
        window_width: Optional[float] = None,
        rescale_slope: float = 1.0,
        rescale_intercept: float = 0.0,
    ) -> "DicomBuilder":
        """Set image data and related parameters."""
        self._dataset.pixel_data = pixel_data

        if pixel_spacing:
            self._dataset.set_value(DicomTags.PixelSpacing, f"{pixel_spacing[0]}\\{pixel_spacing[1]}")
        if slice_thickness:
            self._dataset.set_value(DicomTags.SliceThickness, str(slice_thickness))
        if image_position:
            self._dataset.set_value(
                DicomTags.ImagePositionPatient,
                "\\".join(str(x) for x in image_position)
            )
        if image_orientation:
            self._dataset.set_value(
                DicomTags.ImageOrientationPatient,
                "\\".join(str(x) for x in image_orientation)
            )
        if window_center is not None:
            self._dataset.set_value(DicomTags.WindowCenter, str(window_center))
        if window_width is not None:
            self._dataset.set_value(DicomTags.WindowWidth, str(window_width))

        self._dataset.set_value(DicomTags.RescaleSlope, str(rescale_slope))
        self._dataset.set_value(DicomTags.RescaleIntercept, str(rescale_intercept))

        return self

    def set_sop_class(self, sop_class_uid: str) -> "DicomBuilder":
        """Set SOP Class UID."""
        self._dataset.set_value(DicomTags.SOPClassUID, sop_class_uid)
        return self

    def set_instance_number(self, number: int) -> "DicomBuilder":
        """Set instance number."""
        self._dataset.set_value(DicomTags.InstanceNumber, number)
        return self

    def set_image_type(self, image_type: list[str]) -> "DicomBuilder":
        """Set image type."""
        self._dataset.set_value(DicomTags.ImageType, "\\".join(image_type))
        return self

    def add_custom_element(self, tag: Any, value: Any) -> "DicomBuilder":
        """Add custom element."""
        self._dataset.set_value(tag, value)
        return self

    def build(self) -> DicomDataset:
        """Build and return the DICOM dataset."""
        return self._dataset.copy()

    def reset(self):
        """Reset builder for new dataset."""
        self._dataset = DicomDataset()
        self._set_defaults()


class ModalityBuilder:
    """Specialized builders for specific modalities."""

    @staticmethod
    def create_ultrasound_dataset(
        pixel_data: np.ndarray,
        patient_name: str,
        patient_id: str,
        transducer_type: str = "LINEAR",
        transducer_frequency: int = 7500000,  # Hz
        depth_cm: float = 10.0,
        mechanical_index: float = 1.0,
        thermal_index: float = 1.0,
        manufacturer: str = "",
        model: str = "",
        is_multiframe: bool = False,
        frame_rate: float = 30.0,
    ) -> DicomDataset:
        """Create an ultrasound DICOM dataset."""
        builder = DicomBuilder()

        sop_class = (
            SOPClass.UltrasoundMultiframeImageStorage
            if is_multiframe
            else SOPClass.UltrasoundImageStorage
        )

        builder.set_patient(patient_name, patient_id)
        builder.set_series(modality="US", description="Ultrasound")
        builder.set_equipment(manufacturer=manufacturer, model_name=model)
        builder.set_sop_class(sop_class)
        builder.set_image(pixel_data)
        builder.set_image_type(["ORIGINAL", "PRIMARY"])

        dataset = builder.build()

        # Add ultrasound-specific tags
        dataset.set_value(ModalityTags.US.TransducerType, transducer_type)
        dataset.set_value(ModalityTags.US.TransducerFrequency, transducer_frequency)
        dataset.set_value(ModalityTags.US.DepthOfScanField, int(depth_cm * 10))  # mm
        dataset.set_value(ModalityTags.US.MechanicalIndex, str(mechanical_index))
        dataset.set_value(ModalityTags.US.ThermalIndex, str(thermal_index))

        if is_multiframe and pixel_data.ndim == 3:
            dataset.set_value(ModalityTags.US.NumberOfFrames, pixel_data.shape[0])
            dataset.set_value(ModalityTags.US.FrameTime, str(1000 / frame_rate))

        return dataset

    @staticmethod
    def create_xray_dataset(
        pixel_data: np.ndarray,
        patient_name: str,
        patient_id: str,
        kvp: float = 80.0,
        exposure_time_ms: int = 50,
        tube_current_ma: int = 200,
        sfd_mm: float = 1000.0,
        focal_spot: float = 1.2,
        body_part: str = "CHEST",
        manufacturer: str = "",
        model: str = "",
        pixel_spacing: Optional[tuple[float, float]] = None,
    ) -> DicomDataset:
        """Create a digital X-ray DICOM dataset."""
        builder = DicomBuilder()

        builder.set_patient(patient_name, patient_id)
        builder.set_series(modality="DX", description=f"X-Ray {body_part}", body_part=body_part)
        builder.set_equipment(manufacturer=manufacturer, model_name=model)
        builder.set_sop_class(SOPClass.DigitalXRayImageStorageForPresentation)
        builder.set_image(pixel_data, pixel_spacing=pixel_spacing)
        builder.set_image_type(["ORIGINAL", "PRIMARY"])

        dataset = builder.build()

        # Add X-ray specific tags
        dataset.set_value(ModalityTags.DX.KVP, str(kvp))
        dataset.set_value(ModalityTags.DX.ExposureTime, exposure_time_ms)
        dataset.set_value(ModalityTags.DX.XRayTubeCurrent, tube_current_ma)
        dataset.set_value(ModalityTags.DX.Exposure, int(tube_current_ma * exposure_time_ms / 1000))
        dataset.set_value(ModalityTags.DX.DistanceSourceToDetector, str(sfd_mm))
        dataset.set_value(ModalityTags.DX.FocalSpots, str(focal_spot))
        dataset.set_value(ModalityTags.DX.DetectorType, "DIRECT")

        if pixel_spacing:
            dataset.set_value(ModalityTags.DX.ImagerPixelSpacing, f"{pixel_spacing[0]}\\{pixel_spacing[1]}")

        return dataset

    @staticmethod
    def create_oct_dataset(
        pixel_data: np.ndarray,
        patient_name: str,
        patient_id: str,
        wavelength_nm: float = 840.0,
        bandwidth_nm: float = 50.0,
        axial_resolution_um: float = 5.0,
        lateral_resolution_um: float = 15.0,
        scan_pattern: str = "B_SCAN",
        manufacturer: str = "",
        model: str = "",
        eye_side: str = "R",
    ) -> DicomDataset:
        """Create an OCT DICOM dataset."""
        builder = DicomBuilder()

        builder.set_patient(patient_name, patient_id)
        builder.set_series(modality="OPT", description=f"OCT {eye_side} Eye", body_part="EYE")
        builder.set_equipment(manufacturer=manufacturer, model_name=model)
        builder.set_sop_class(SOPClass.OphthalmicTomographyImageStorage)
        builder.set_image(pixel_data)
        builder.set_image_type(["ORIGINAL", "PRIMARY"])

        dataset = builder.build()

        # Add OCT-specific tags
        dataset.set_value(ModalityTags.OPT.IlluminationWaveLength, wavelength_nm)
        dataset.set_value(ModalityTags.OPT.IlluminationBandwidth, bandwidth_nm)
        dataset.set_value(ModalityTags.OPT.DepthSpatialResolution, axial_resolution_um)

        return dataset

    @staticmethod
    def create_thermal_dataset(
        pixel_data: np.ndarray,
        patient_name: str,
        patient_id: str,
        min_temp: float,
        max_temp: float,
        emissivity: float = 0.98,
        ambient_temp: float = 22.0,
        distance_m: float = 1.0,
        sensor_type: str = "MICROBOLOMETER",
        manufacturer: str = "",
        model: str = "",
        body_part: str = "",
    ) -> DicomDataset:
        """Create a thermal imaging DICOM dataset (as secondary capture)."""
        builder = DicomBuilder()

        builder.set_patient(patient_name, patient_id)
        builder.set_series(modality="SC", description=f"Thermal Imaging {body_part}", body_part=body_part)
        builder.set_equipment(manufacturer=manufacturer, model_name=model)
        builder.set_sop_class(SOPClass.SecondaryCaptureImageStorage)
        builder.set_image(pixel_data)
        builder.set_image_type(["DERIVED", "SECONDARY"])

        dataset = builder.build()

        # Add thermal-specific private tags
        dataset.set_value(ModalityTags.TG.PrivateCreator, "THERMAL_IMAGING")
        dataset.set_value(ModalityTags.TG.SensorType, sensor_type)
        dataset.set_value(ModalityTags.TG.Emissivity, str(emissivity))
        dataset.set_value(ModalityTags.TG.AmbientTemperature, str(ambient_temp))
        dataset.set_value(ModalityTags.TG.MinTemperature, str(min_temp))
        dataset.set_value(ModalityTags.TG.MaxTemperature, str(max_temp))
        dataset.set_value(ModalityTags.TG.DistanceToTarget, str(distance_m))

        return dataset


class StudyBuilder:
    """Builder for creating complete DICOM studies."""

    def __init__(
        self,
        patient_name: str,
        patient_id: str,
        study_description: str = "",
    ):
        self._study = DicomStudy(
            study_instance_uid=generate_uid("study"),
            patient_name=patient_name,
            patient_id=patient_id,
            study_description=study_description,
            study_date=datetime.now().date(),
            study_time=datetime.now().time(),
        )

    def set_patient_info(
        self,
        birth_date: Optional[date] = None,
        sex: str = "",
    ) -> "StudyBuilder":
        """Set additional patient information."""
        self._study.patient_birth_date = birth_date
        self._study.patient_sex = sex
        return self

    def set_study_info(
        self,
        accession_number: str = "",
        referring_physician: str = "",
        study_id: str = "",
    ) -> "StudyBuilder":
        """Set study information."""
        self._study.accession_number = accession_number
        self._study.referring_physician = referring_physician
        self._study.study_id = study_id
        return self

    def add_ultrasound_series(
        self,
        images: list[np.ndarray],
        description: str = "Ultrasound",
        body_part: str = "",
        **kwargs,
    ) -> "StudyBuilder":
        """Add ultrasound series to study."""
        series = self._study.create_series("US", description, body_part)
        for i, img in enumerate(images):
            dataset = ModalityBuilder.create_ultrasound_dataset(
                img,
                self._study.patient_name,
                self._study.patient_id,
                **kwargs,
            )
            dataset.set_value(DicomTags.InstanceNumber, i + 1)
            series.instances.append(dataset)
        return self

    def add_xray_series(
        self,
        images: list[np.ndarray],
        description: str = "X-Ray",
        body_part: str = "CHEST",
        **kwargs,
    ) -> "StudyBuilder":
        """Add X-ray series to study."""
        series = self._study.create_series("DX", description, body_part)
        for i, img in enumerate(images):
            dataset = ModalityBuilder.create_xray_dataset(
                img,
                self._study.patient_name,
                self._study.patient_id,
                body_part=body_part,
                **kwargs,
            )
            dataset.set_value(DicomTags.InstanceNumber, i + 1)
            series.instances.append(dataset)
        return self

    def add_oct_series(
        self,
        images: list[np.ndarray],
        description: str = "OCT",
        **kwargs,
    ) -> "StudyBuilder":
        """Add OCT series to study."""
        series = self._study.create_series("OPT", description, "EYE")
        for i, img in enumerate(images):
            dataset = ModalityBuilder.create_oct_dataset(
                img,
                self._study.patient_name,
                self._study.patient_id,
                **kwargs,
            )
            dataset.set_value(DicomTags.InstanceNumber, i + 1)
            series.instances.append(dataset)
        return self

    def add_thermal_series(
        self,
        images: list[np.ndarray],
        temperature_ranges: list[tuple[float, float]],
        description: str = "Thermal",
        body_part: str = "",
        **kwargs,
    ) -> "StudyBuilder":
        """Add thermal series to study."""
        series = self._study.create_series("SC", description, body_part)
        for i, (img, (min_t, max_t)) in enumerate(zip(images, temperature_ranges)):
            dataset = ModalityBuilder.create_thermal_dataset(
                img,
                self._study.patient_name,
                self._study.patient_id,
                min_temp=min_t,
                max_temp=max_t,
                body_part=body_part,
                **kwargs,
            )
            dataset.set_value(DicomTags.InstanceNumber, i + 1)
            series.instances.append(dataset)
        return self

    def build(self) -> DicomStudy:
        """Build and return the study."""
        # Update all datasets with study-level tags
        for series in self._study.series:
            for dataset in series.instances:
                self._study._set_study_tags(dataset)
        return self._study
