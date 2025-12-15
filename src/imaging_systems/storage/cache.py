"""Caching layer for medical images."""

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any
import threading
import numpy as np


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    data: Any
    size_bytes: int
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0


class ImageCache(ABC):
    """Abstract cache interface."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def put(self, key: str, data: Any) -> bool:
        pass

    @abstractmethod
    def remove(self, key: str) -> bool:
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def contains(self, key: str) -> bool:
        pass


class LRUCache(ImageCache):
    """Least Recently Used cache implementation."""

    def __init__(
        self,
        max_size_mb: float = 1024,
        max_items: int = 1000,
    ):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.max_items = max_items

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._current_size = 0
        self._lock = threading.RLock()

        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry = self._cache[key]
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                self._hits += 1
                return entry.data
            else:
                self._misses += 1
                return None

    def put(self, key: str, data: Any) -> bool:
        with self._lock:
            # Calculate size
            if isinstance(data, np.ndarray):
                size = data.nbytes
            elif isinstance(data, bytes):
                size = len(data)
            else:
                size = 1024  # Default estimate

            # Check if we need to evict
            while (self._current_size + size > self.max_size_bytes or
                   len(self._cache) >= self.max_items):
                if not self._cache:
                    break
                self._evict_one()

            # Remove if exists
            if key in self._cache:
                old_entry = self._cache.pop(key)
                self._current_size -= old_entry.size_bytes

            # Add new entry
            entry = CacheEntry(
                key=key,
                data=data,
                size_bytes=size,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
            )
            self._cache[key] = entry
            self._current_size += size

            return True

    def _evict_one(self):
        """Evict the least recently used item."""
        if self._cache:
            # First item is LRU
            key, entry = self._cache.popitem(last=False)
            self._current_size -= entry.size_bytes

    def remove(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                entry = self._cache.pop(key)
                self._current_size -= entry.size_bytes
                return True
            return False

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._current_size = 0
            self._hits = 0
            self._misses = 0

    def contains(self, key: str) -> bool:
        return key in self._cache

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        return {
            "items": len(self._cache),
            "size_bytes": self._current_size,
            "size_mb": self._current_size / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }


class TieredCache(ImageCache):
    """Two-tier cache (memory + disk)."""

    def __init__(
        self,
        memory_cache: ImageCache,
        disk_cache: ImageCache,
    ):
        self._memory_cache = memory_cache
        self._disk_cache = disk_cache

    def get(self, key: str) -> Optional[Any]:
        # Try memory first
        data = self._memory_cache.get(key)
        if data is not None:
            return data

        # Try disk
        data = self._disk_cache.get(key)
        if data is not None:
            # Promote to memory cache
            self._memory_cache.put(key, data)
            return data

        return None

    def put(self, key: str, data: Any) -> bool:
        # Put in both caches
        self._memory_cache.put(key, data)
        self._disk_cache.put(key, data)
        return True

    def remove(self, key: str) -> bool:
        m = self._memory_cache.remove(key)
        d = self._disk_cache.remove(key)
        return m or d

    def clear(self):
        self._memory_cache.clear()
        self._disk_cache.clear()

    def contains(self, key: str) -> bool:
        return self._memory_cache.contains(key) or self._disk_cache.contains(key)


class DiskCache(ImageCache):
    """Disk-based cache."""

    def __init__(
        self,
        cache_dir: str,
        max_size_mb: float = 10240,
    ):
        from pathlib import Path
        import json

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self._index_file = self.cache_dir / "cache_index.json"
        self._index: dict[str, dict] = {}
        self._load_index()

    def _load_index(self):
        """Load cache index from disk."""
        import json
        if self._index_file.exists():
            with open(self._index_file) as f:
                self._index = json.load(f)

    def _save_index(self):
        """Save cache index to disk."""
        import json
        with open(self._index_file, "w") as f:
            json.dump(self._index, f)

    def _get_cache_path(self, key: str) -> "Path":
        import hashlib
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def get(self, key: str) -> Optional[Any]:
        if key not in self._index:
            return None

        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            del self._index[key]
            self._save_index()
            return None

        with open(cache_path, "rb") as f:
            data = f.read()

        # Update access time
        self._index[key]["last_accessed"] = datetime.now().isoformat()
        self._save_index()

        return data

    def put(self, key: str, data: Any) -> bool:
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        elif not isinstance(data, bytes):
            data = str(data).encode()

        cache_path = self._get_cache_path(key)

        with open(cache_path, "wb") as f:
            f.write(data)

        self._index[key] = {
            "size": len(data),
            "created": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
        }
        self._save_index()

        return True

    def remove(self, key: str) -> bool:
        if key not in self._index:
            return False

        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()

        del self._index[key]
        self._save_index()
        return True

    def clear(self):
        import shutil
        for f in self.cache_dir.glob("*.cache"):
            f.unlink()
        self._index.clear()
        self._save_index()

    def contains(self, key: str) -> bool:
        return key in self._index and self._get_cache_path(key).exists()
