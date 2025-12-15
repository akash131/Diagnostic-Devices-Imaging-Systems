"""Image storage implementations."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import gzip
import hashlib
import shutil
import numpy as np


class ImageStore(ABC):
    """Abstract image storage interface."""

    @abstractmethod
    def store(self, image_id: str, data: bytes, metadata: dict = None) -> str:
        """Store image data and return path."""
        pass

    @abstractmethod
    def retrieve(self, image_id: str) -> Optional[bytes]:
        """Retrieve image data by ID."""
        pass

    @abstractmethod
    def delete(self, image_id: str) -> bool:
        """Delete image data."""
        pass

    @abstractmethod
    def exists(self, image_id: str) -> bool:
        """Check if image exists."""
        pass


class FileSystemStore(ImageStore):
    """File system-based image storage."""

    def __init__(
        self,
        base_path: str,
        use_subdirs: bool = True,
        subdir_depth: int = 2,
    ):
        self.base_path = Path(base_path)
        self.use_subdirs = use_subdirs
        self.subdir_depth = subdir_depth
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, image_id: str) -> Path:
        """Generate file path for image ID."""
        # Hash the ID to generate subdirectory structure
        id_hash = hashlib.md5(image_id.encode()).hexdigest()

        if self.use_subdirs:
            subdir_parts = [id_hash[i:i + 2] for i in range(0, self.subdir_depth * 2, 2)]
            subdir = self.base_path.joinpath(*subdir_parts)
        else:
            subdir = self.base_path

        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{image_id}.dcm"

    def store(self, image_id: str, data: bytes, metadata: dict = None) -> str:
        file_path = self._get_file_path(image_id)

        with open(file_path, "wb") as f:
            f.write(data)

        return str(file_path)

    def retrieve(self, image_id: str) -> Optional[bytes]:
        file_path = self._get_file_path(image_id)

        if not file_path.exists():
            return None

        with open(file_path, "rb") as f:
            return f.read()

    def delete(self, image_id: str) -> bool:
        file_path = self._get_file_path(image_id)

        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def exists(self, image_id: str) -> bool:
        return self._get_file_path(image_id).exists()

    def get_storage_stats(self) -> dict:
        """Get storage statistics."""
        total_files = 0
        total_size = 0

        for file in self.base_path.rglob("*"):
            if file.is_file():
                total_files += 1
                total_size += file.stat().st_size

        return {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "base_path": str(self.base_path),
        }


class CompressedStore(ImageStore):
    """Compressed image storage using gzip."""

    def __init__(
        self,
        base_path: str,
        compression_level: int = 6,
    ):
        self.base_path = Path(base_path)
        self.compression_level = compression_level
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, image_id: str) -> Path:
        id_hash = hashlib.md5(image_id.encode()).hexdigest()
        subdir = self.base_path / id_hash[:2] / id_hash[2:4]
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{image_id}.dcm.gz"

    def store(self, image_id: str, data: bytes, metadata: dict = None) -> str:
        file_path = self._get_file_path(image_id)

        with gzip.open(file_path, "wb", compresslevel=self.compression_level) as f:
            f.write(data)

        return str(file_path)

    def retrieve(self, image_id: str) -> Optional[bytes]:
        file_path = self._get_file_path(image_id)

        if not file_path.exists():
            return None

        with gzip.open(file_path, "rb") as f:
            return f.read()

    def delete(self, image_id: str) -> bool:
        file_path = self._get_file_path(image_id)

        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def exists(self, image_id: str) -> bool:
        return self._get_file_path(image_id).exists()


class NumpyStore(ImageStore):
    """Storage for numpy arrays (raw pixel data)."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, image_id: str) -> Path:
        return self.base_path / f"{image_id}.npy"

    def store(self, image_id: str, data: bytes, metadata: dict = None) -> str:
        # Data should be serialized numpy array
        file_path = self._get_file_path(image_id)
        with open(file_path, "wb") as f:
            f.write(data)
        return str(file_path)

    def store_array(self, image_id: str, array: np.ndarray, compress: bool = True) -> str:
        """Store numpy array directly."""
        file_path = self._get_file_path(image_id)

        if compress:
            file_path = file_path.with_suffix(".npz")
            np.savez_compressed(file_path, data=array)
        else:
            np.save(file_path, array)

        return str(file_path)

    def retrieve(self, image_id: str) -> Optional[bytes]:
        file_path = self._get_file_path(image_id)

        if not file_path.exists():
            # Try compressed version
            file_path = file_path.with_suffix(".npz")
            if not file_path.exists():
                return None

        with open(file_path, "rb") as f:
            return f.read()

    def retrieve_array(self, image_id: str) -> Optional[np.ndarray]:
        """Retrieve as numpy array."""
        file_path = self._get_file_path(image_id)

        if file_path.exists():
            return np.load(file_path)

        # Try compressed
        file_path = file_path.with_suffix(".npz")
        if file_path.exists():
            with np.load(file_path) as data:
                return data["data"]

        return None

    def delete(self, image_id: str) -> bool:
        file_path = self._get_file_path(image_id)

        deleted = False
        if file_path.exists():
            file_path.unlink()
            deleted = True

        compressed_path = file_path.with_suffix(".npz")
        if compressed_path.exists():
            compressed_path.unlink()
            deleted = True

        return deleted

    def exists(self, image_id: str) -> bool:
        file_path = self._get_file_path(image_id)
        return file_path.exists() or file_path.with_suffix(".npz").exists()


class HierarchicalStore(ImageStore):
    """Hierarchical storage organized by study/series/instance."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _parse_hierarchical_id(self, image_id: str) -> tuple[str, str, str]:
        """Parse image_id in format study_uid/series_uid/instance_uid."""
        parts = image_id.split("/")
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        return "", "", image_id

    def _get_file_path(self, image_id: str) -> Path:
        study_uid, series_uid, instance_uid = self._parse_hierarchical_id(image_id)

        if study_uid and series_uid:
            path = self.base_path / study_uid[:8] / series_uid[:8]
        else:
            path = self.base_path / "unorganized"

        path.mkdir(parents=True, exist_ok=True)
        return path / f"{instance_uid}.dcm"

    def store(self, image_id: str, data: bytes, metadata: dict = None) -> str:
        file_path = self._get_file_path(image_id)

        with open(file_path, "wb") as f:
            f.write(data)

        return str(file_path)

    def retrieve(self, image_id: str) -> Optional[bytes]:
        file_path = self._get_file_path(image_id)

        if not file_path.exists():
            return None

        with open(file_path, "rb") as f:
            return f.read()

    def delete(self, image_id: str) -> bool:
        file_path = self._get_file_path(image_id)

        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def exists(self, image_id: str) -> bool:
        return self._get_file_path(image_id).exists()

    def delete_study(self, study_uid: str) -> bool:
        """Delete all files for a study."""
        study_path = self.base_path / study_uid[:8]
        if study_path.exists():
            shutil.rmtree(study_path)
            return True
        return False
