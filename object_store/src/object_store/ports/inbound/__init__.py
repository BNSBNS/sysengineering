"""Inbound ports - API contracts for the object store.

Inbound ports define the interfaces that clients and upper layers
use to interact with object storage and chunk management.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol

from object_store.domain.entities.bucket import Bucket
from object_store.domain.entities.chunk import Chunk, ChunkRef
from object_store.domain.entities.object import Object, ObjectVersion


# =============================================================================
# Object Service Port
# =============================================================================


@dataclass
class ObjectServiceStats:
    """Statistics for object service monitoring."""

    total_objects: int
    total_chunks: int
    total_size_bytes: int
    integrity_checks_passed: int
    integrity_checks_failed: int


class ObjectServicePort(Protocol):
    """Protocol for object storage operations.

    Handles S3-compatible object operations with content-addressable storage.

    Thread Safety:
        All methods must be thread-safe.

    Integrity:
        All objects have SHA-256 checksums for corruption detection.

    Example:
        obj = service.put_object(bucket, "my-file.txt", data)
        retrieved = service.get_object(obj.object_id)
        if service.verify_object_integrity(obj.object_id):
            # Object is intact
            pass
    """

    @abstractmethod
    def put_object(
        self,
        bucket: Bucket,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> Object:
        """Put an object in storage.

        Args:
            bucket: Target bucket.
            key: Object key.
            data: Object data.
            content_type: MIME type.

        Returns:
            Stored object.
        """
        ...

    @abstractmethod
    def get_object(self, object_id: str) -> Optional[Object]:
        """Get object metadata.

        Args:
            object_id: Object ID.

        Returns:
            Object or None if not found.
        """
        ...

    @abstractmethod
    def get_object_data(self, object_id: str) -> Optional[bytes]:
        """Get object data (reassembled from chunks).

        Args:
            object_id: Object ID.

        Returns:
            Object data or None if not found.
        """
        ...

    @abstractmethod
    def delete_object(self, object_id: str) -> bool:
        """Delete an object and its chunks.

        Args:
            object_id: Object ID.

        Returns:
            True if deleted.
        """
        ...

    @abstractmethod
    def list_objects(self, bucket: str, prefix: str = "") -> list[Object]:
        """List objects in a bucket.

        Args:
            bucket: Bucket name.
            prefix: Key prefix filter.

        Returns:
            List of objects.
        """
        ...

    @abstractmethod
    def verify_object_integrity(self, object_id: str) -> bool:
        """Verify object integrity via chunk checksums.

        Args:
            object_id: Object ID.

        Returns:
            True if all chunks are valid.
        """
        ...

    @abstractmethod
    def get_stats(self) -> ObjectServiceStats:
        """Get object service statistics.

        Returns:
            Object service statistics.
        """
        ...


# =============================================================================
# Bucket Service Port
# =============================================================================


@dataclass
class BucketServiceStats:
    """Statistics for bucket service monitoring."""

    total_buckets: int
    total_objects: int
    total_size_bytes: int


class BucketServicePort(Protocol):
    """Protocol for bucket management operations.

    Handles bucket lifecycle and configuration.

    Thread Safety:
        All methods must be thread-safe.

    Example:
        bucket = service.create_bucket("my-bucket")
        service.set_versioning(bucket.name, enabled=True)
        buckets = service.list_buckets()
    """

    @abstractmethod
    def create_bucket(
        self,
        name: str,
        versioning_enabled: bool = False,
        compression_enabled: bool = False,
    ) -> Bucket:
        """Create a new bucket.

        Args:
            name: Bucket name.
            versioning_enabled: Enable object versioning.
            compression_enabled: Enable compression.

        Returns:
            Created bucket.

        Raises:
            BucketError: If bucket already exists.
        """
        ...

    @abstractmethod
    def get_bucket(self, name: str) -> Optional[Bucket]:
        """Get bucket by name.

        Args:
            name: Bucket name.

        Returns:
            Bucket or None if not found.
        """
        ...

    @abstractmethod
    def delete_bucket(self, name: str) -> bool:
        """Delete a bucket.

        Bucket must be empty.

        Args:
            name: Bucket name.

        Returns:
            True if deleted.

        Raises:
            BucketError: If bucket is not empty.
        """
        ...

    @abstractmethod
    def list_buckets(self) -> list[Bucket]:
        """List all buckets.

        Returns:
            List of buckets.
        """
        ...

    @abstractmethod
    def set_versioning(self, name: str, enabled: bool) -> None:
        """Enable or disable versioning for a bucket.

        Args:
            name: Bucket name.
            enabled: Whether to enable versioning.
        """
        ...

    @abstractmethod
    def get_stats(self) -> BucketServiceStats:
        """Get bucket service statistics.

        Returns:
            Bucket service statistics.
        """
        ...


class BucketError(Exception):
    """Raised when bucket operation fails."""

    pass


# =============================================================================
# Chunk Store Port
# =============================================================================


@dataclass
class ChunkStoreStats:
    """Statistics for chunk store monitoring."""

    total_chunks: int
    total_size_bytes: int
    deduplication_ratio: float


class ChunkStorePort(Protocol):
    """Protocol for content-addressable chunk storage.

    Handles chunk storage with deduplication and integrity verification.

    Thread Safety:
        All methods must be thread-safe.

    Deduplication:
        Chunks are addressed by SHA-256 hash for automatic deduplication.

    Example:
        chunk = store.put_chunk(data)
        retrieved = store.get_chunk(chunk.chunk_id)
        if store.verify_chunk(chunk.chunk_id):
            # Chunk is intact
            pass
    """

    @abstractmethod
    def put_chunk(self, data: bytes, object_id: str, sequence: int) -> Chunk:
        """Store a chunk.

        Args:
            data: Chunk data.
            object_id: Parent object ID.
            sequence: Chunk sequence number.

        Returns:
            Stored chunk.
        """
        ...

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Get a chunk by ID.

        Args:
            chunk_id: Chunk ID.

        Returns:
            Chunk or None if not found.
        """
        ...

    @abstractmethod
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk.

        Args:
            chunk_id: Chunk ID.

        Returns:
            True if deleted.
        """
        ...

    @abstractmethod
    def verify_chunk(self, chunk_id: str) -> bool:
        """Verify chunk integrity via checksum.

        Args:
            chunk_id: Chunk ID.

        Returns:
            True if chunk is intact.
        """
        ...

    @abstractmethod
    def get_stats(self) -> ChunkStoreStats:
        """Get chunk store statistics.

        Returns:
            Chunk store statistics.
        """
        ...


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Object Service
    "ObjectServicePort",
    "ObjectServiceStats",
    # Bucket Service
    "BucketServicePort",
    "BucketServiceStats",
    "BucketError",
    # Chunk Store
    "ChunkStorePort",
    "ChunkStoreStats",
]
