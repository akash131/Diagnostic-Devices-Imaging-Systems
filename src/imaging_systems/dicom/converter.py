"""DICOM conversion utilities for various image formats."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import numpy as np

from .dataset import DicomDataset, DicomSeries, DicomStudy
from .builder import DicomBuilder, ModalityBuilder
from .tags import DicomTags
from ..base import ImageData


@dataclass
class ConversionOptions:
    """Options for image conversion."""

    rescale_to_bits: int = 16
    apply_windowing: bool = False
    window_center: Optional[float] = None
    window_width: Optional[float] = None
    normalize: bool = False
    compress: bool = False


class DicomConverter:
    """Convert between DICOM and other formats."""

    def __init__(self, options: Optional[ConversionOptions] = None):
        self.options = options or ConversionOptions()

    def to_numpy(self, dataset: DicomDataset) -> np.ndarray:
        """Convert DICOM dataset to numpy array."""
        if dataset.pixel_data is None:
            raise ValueError("No pixel data in dataset")

        image = dataset.pixel_data.copy()

        # Apply rescale
        slope = float(dataset.get_value(DicomTags.RescaleSlope, 1.0))
        intercept = float(dataset.get_value(DicomTags.RescaleIntercept, 0.0))

        if slope != 1.0 or intercept != 0.0:
            image = image.astype(np.float64) * slope + intercept

        # Apply windowing if requested
        if self.options.apply_windowing:
            wc = self.options.window_center or dataset.get_value(DicomTags.WindowCenter)
            ww = self.options.window_width or dataset.get_value(DicomTags.WindowWidth)

            if wc is not None and ww is not None:
                wc = float(wc.split("\\")[0]) if isinstance(wc, str) else float(wc)
                ww = float(ww.split("\\")[0]) if isinstance(ww, str) else float(ww)
                image = self._apply_windowing(image, wc, ww)

        # Normalize if requested
        if self.options.normalize:
            image = self._normalize(image)

        return image

    def _apply_windowing(
        self,
        image: np.ndarray,
        window_center: float,
        window_width: float,
    ) -> np.ndarray:
        """Apply window/level to image."""
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2

        image = np.clip(image, min_val, max_val)
        image = (image - min_val) / (max_val - min_val) * 255

        return image.astype(np.uint8)

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-1 range."""
        min_val = image.min()
        max_val = image.max()
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        return image

    def to_png(self, dataset: DicomDataset, output_path: str | Path):
        """Convert DICOM to PNG."""
        from PIL import Image

        image = self.to_numpy(dataset)

        # Ensure 8-bit for PNG
        if image.dtype != np.uint8:
            if image.max() > 255:
                image = (image / image.max() * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        pil_image = Image.fromarray(image)
        pil_image.save(output_path)

    def to_tiff(
        self,
        dataset: DicomDataset,
        output_path: str | Path,
        preserve_bits: bool = True,
    ):
        """Convert DICOM to TIFF (preserves bit depth)."""
        from PIL import Image

        image = self.to_numpy(dataset)

        if not preserve_bits:
            if image.dtype != np.uint8:
                image = (image / image.max() * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)
        pil_image.save(output_path, format="TIFF")

    def to_nifti(self, datasets: list[DicomDataset], output_path: str | Path):
        """Convert DICOM series to NIfTI format."""
        # Stack images into 3D volume
        volume = np.stack([d.pixel_data for d in datasets if d.pixel_data is not None])

        # Get spacing information
        pixel_spacing = datasets[0].get_value(DicomTags.PixelSpacing, "1\\1")
        slice_thickness = float(datasets[0].get_value(DicomTags.SliceThickness, 1.0))

        if isinstance(pixel_spacing, str):
            ps = [float(x) for x in pixel_spacing.split("\\")]
        else:
            ps = [1.0, 1.0]

        # Create affine matrix
        affine = np.eye(4)
        affine[0, 0] = ps[0]
        affine[1, 1] = ps[1]
        affine[2, 2] = slice_thickness

        # Save using nibabel-like format (simplified)
        np.savez(
            output_path,
            data=volume,
            affine=affine,
            pixel_spacing=ps,
            slice_thickness=slice_thickness,
        )

    def series_to_numpy(self, series: DicomSeries) -> np.ndarray:
        """Convert DICOM series to 3D numpy array."""
        images = []
        for dataset in series.instances:
            if dataset.pixel_data is not None:
                images.append(self.to_numpy(dataset))

        if not images:
            raise ValueError("No images in series")

        return np.stack(images)


class ImageToDicom:
    """Convert various image formats to DICOM."""

    def __init__(
        self,
        patient_name: str = "Anonymous",
        patient_id: str = "000000",
        modality: str = "SC",
    ):
        self.patient_name = patient_name
        self.patient_id = patient_id
        self.modality = modality

    def from_numpy(
        self,
        image: np.ndarray,
        pixel_spacing: Optional[tuple[float, float]] = None,
        description: str = "",
        **kwargs,
    ) -> DicomDataset:
        """Convert numpy array to DICOM dataset."""
        builder = DicomBuilder()

        builder.set_patient(self.patient_name, self.patient_id)
        builder.set_series(modality=self.modality, description=description)
        builder.set_image(image, pixel_spacing=pixel_spacing)

        # Add any additional parameters
        for key, value in kwargs.items():
            if hasattr(DicomTags, key):
                builder.add_custom_element(getattr(DicomTags, key), value)

        return builder.build()

    def from_png(self, file_path: str | Path, **kwargs) -> DicomDataset:
        """Convert PNG to DICOM dataset."""
        from PIL import Image

        pil_image = Image.open(file_path)
        image = np.array(pil_image)

        return self.from_numpy(image, **kwargs)

    def from_tiff(self, file_path: str | Path, **kwargs) -> DicomDataset:
        """Convert TIFF to DICOM dataset."""
        from PIL import Image

        pil_image = Image.open(file_path)
        image = np.array(pil_image)

        return self.from_numpy(image, **kwargs)

    def from_imaging_data(
        self,
        image_data: ImageData,
        **kwargs,
    ) -> DicomDataset:
        """Convert ImageData object to DICOM dataset."""
        modality_map = {
            "ultrasound": "US",
            "xray": "DX",
            "oct": "OPT",
            "thermal": "SC",
        }

        modality = modality_map.get(image_data.modality, self.modality)

        if image_data.modality == "ultrasound":
            return ModalityBuilder.create_ultrasound_dataset(
                image_data.data,
                self.patient_name,
                self.patient_id,
                **{**image_data.metadata, **kwargs},
            )
        elif image_data.modality == "xray":
            return ModalityBuilder.create_xray_dataset(
                image_data.data,
                self.patient_name,
                self.patient_id,
                **{**image_data.metadata, **kwargs},
            )
        elif image_data.modality == "oct":
            return ModalityBuilder.create_oct_dataset(
                image_data.data,
                self.patient_name,
                self.patient_id,
                **{**image_data.metadata, **kwargs},
            )
        elif image_data.modality == "thermal":
            min_temp = image_data.metadata.get("min_temp_c", image_data.data.min())
            max_temp = image_data.metadata.get("max_temp_c", image_data.data.max())
            return ModalityBuilder.create_thermal_dataset(
                image_data.data,
                self.patient_name,
                self.patient_id,
                min_temp=min_temp,
                max_temp=max_temp,
                **{k: v for k, v in image_data.metadata.items() if k not in ["min_temp_c", "max_temp_c"]},
                **kwargs,
            )
        else:
            return self.from_numpy(
                image_data.data,
                description=image_data.modality,
                **kwargs,
            )


class DicomToImage:
    """Convert DICOM to ImageData format."""

    def to_image_data(self, dataset: DicomDataset) -> ImageData:
        """Convert DICOM dataset to ImageData."""
        converter = DicomConverter()
        image = converter.to_numpy(dataset)

        modality = dataset.get_value(DicomTags.Modality, "")
        modality_map = {
            "US": "ultrasound",
            "DX": "xray",
            "CR": "xray",
            "OPT": "oct",
            "SC": "secondary_capture",
        }

        # Build metadata from DICOM tags
        metadata = {}
        for element in dataset:
            if element.value is not None:
                metadata[element.tag.keyword] = element.value

        return ImageData(
            data=image,
            timestamp=datetime.now(),
            metadata=metadata,
            device_id=dataset.get_value(DicomTags.DeviceSerialNumber, ""),
            modality=modality_map.get(modality, modality.lower()),
        )

    def series_to_image_data_list(self, series: DicomSeries) -> list[ImageData]:
        """Convert DICOM series to list of ImageData."""
        return [self.to_image_data(ds) for ds in series.instances]


class BatchConverter:
    """Batch conversion utilities."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_study_to_png(self, study: DicomStudy) -> list[Path]:
        """Convert all images in study to PNG files."""
        converter = DicomConverter(ConversionOptions(apply_windowing=True))
        output_files = []

        for series in study.series:
            series_dir = self.output_dir / f"series_{series.series_number}"
            series_dir.mkdir(exist_ok=True)

            for i, dataset in enumerate(series.instances):
                output_path = series_dir / f"image_{i + 1:04d}.png"
                converter.to_png(dataset, output_path)
                output_files.append(output_path)

        return output_files

    def convert_directory_to_dicom(
        self,
        input_dir: str | Path,
        patient_name: str,
        patient_id: str,
        modality: str = "SC",
    ) -> DicomStudy:
        """Convert all images in directory to DICOM study."""
        from .builder import StudyBuilder

        input_dir = Path(input_dir)
        builder = StudyBuilder(patient_name, patient_id, f"Converted from {input_dir.name}")

        image_converter = ImageToDicom(patient_name, patient_id, modality)

        # Find all image files
        image_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
        image_files = sorted([
            f for f in input_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ])

        if not image_files:
            raise ValueError(f"No image files found in {input_dir}")

        # Create series and add images
        series = builder._study.create_series(modality, "Converted Images")

        for i, img_path in enumerate(image_files):
            dataset = image_converter.from_png(img_path)
            dataset.set_value(DicomTags.InstanceNumber, i + 1)
            series.instances.append(dataset)

        return builder.build()
