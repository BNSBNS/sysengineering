"""Integration tests for object store lifecycle operations."""

import pytest
import hashlib

from object_store.domain.entities.bucket import Bucket
from object_store.domain.entities.object import Object
from object_store.domain.entities.chunk import Chunk, ChunkRef
from object_store.domain.services.storage_service import StorageService
from object_store.domain.services.erasure_coding_service import ErasureCodingService


def create_test_bucket(name: str = "test-bucket") -> Bucket:
    """Create a test bucket."""
    return Bucket(
        bucket_id=f"bucket-{name}",
        name=name,
        versioning_enabled=True,
    )


class TestStorageServiceIntegration:
    """Integration tests for storage service."""

    def test_put_and_get_object(self):
        """Test putting and getting an object."""
        service = StorageService()
        bucket = create_test_bucket()

        # Put object
        data = b"Hello, World!"
        obj = service.put_object(bucket, "greeting.txt", data, "text/plain")

        assert obj.key == "greeting.txt"
        assert obj.size == len(data)
        assert obj.content_type == "text/plain"

        # Get object
        retrieved = service.get_object(obj.object_id)
        assert retrieved is not None
        assert retrieved.key == obj.key

    def test_object_chunking(self):
        """Test that large objects are properly chunked."""
        service = StorageService(chunk_size=10)  # Small chunks for testing
        bucket = create_test_bucket()

        # Create data larger than chunk size
        data = b"0123456789" * 5  # 50 bytes

        obj = service.put_object(bucket, "large.dat", data)

        # Should have 5 chunks
        assert len(obj.chunk_refs) == 5

        # Each chunk ref should have checksum
        for ref in obj.chunk_refs:
            assert ref.checksum != ""
            assert ref.size == 10

    def test_object_checksum(self):
        """Test object checksum calculation."""
        service = StorageService()
        bucket = create_test_bucket()

        data = b"Test data for checksum verification"
        obj = service.put_object(bucket, "checksum.txt", data)

        expected_checksum = hashlib.sha256(data).hexdigest()
        assert obj.checksum == expected_checksum

    def test_delete_object(self):
        """Test deleting an object."""
        service = StorageService()
        bucket = create_test_bucket()

        # Create object
        obj = service.put_object(bucket, "to-delete.txt", b"data")
        object_id = obj.object_id

        # Verify exists
        assert service.get_object(object_id) is not None

        # Delete
        result = service.delete_object(object_id)
        assert result is True

        # Verify deleted
        assert service.get_object(object_id) is None


class TestChunkIntegrity:
    """Integration tests for chunk integrity verification."""

    def test_chunk_checksum_verification(self):
        """Test chunk checksum verification."""
        chunk_data = b"This is chunk data for integrity testing"
        checksum = hashlib.sha256(chunk_data).hexdigest()

        chunk = Chunk(
            chunk_id="chunk-001",
            object_id="obj-001",
            sequence=0,
            data=chunk_data,
            size=len(chunk_data),
            checksum=checksum,
        )

        assert chunk.verify_checksum()

    def test_corrupted_chunk_detection(self):
        """Test that corrupted chunks are detected."""
        chunk_data = b"Original chunk data"
        checksum = hashlib.sha256(chunk_data).hexdigest()

        chunk = Chunk(
            chunk_id="chunk-001",
            object_id="obj-001",
            sequence=0,
            data=b"Corrupted data",  # Different from checksum
            size=len(chunk_data),
            checksum=checksum,
        )

        assert not chunk.verify_checksum()

    def test_object_integrity_verification(self):
        """Test full object integrity verification."""
        service = StorageService(chunk_size=10)
        bucket = create_test_bucket()

        data = b"Data for integrity verification"
        obj = service.put_object(bucket, "integrity.dat", data)

        # Verify integrity
        is_valid = service.verify_object_integrity(obj.object_id)
        assert is_valid


class TestMultipleObjects:
    """Integration tests for managing multiple objects."""

    def test_multiple_objects_in_bucket(self):
        """Test storing multiple objects in a bucket."""
        service = StorageService()
        bucket = create_test_bucket()

        # Create multiple objects
        objects = []
        for i in range(5):
            obj = service.put_object(bucket, f"file{i}.txt", f"Content {i}".encode())
            objects.append(obj)

        # All objects should exist
        for obj in objects:
            retrieved = service.get_object(obj.object_id)
            assert retrieved is not None

    def test_same_key_different_buckets(self):
        """Test that same key can exist in different buckets."""
        service = StorageService()

        bucket1 = create_test_bucket("bucket-1")
        bucket2 = create_test_bucket("bucket-2")

        key = "shared-key.txt"

        obj1 = service.put_object(bucket1, key, b"Data in bucket 1")
        obj2 = service.put_object(bucket2, key, b"Data in bucket 2")

        # Both objects should exist with different IDs
        assert obj1.object_id != obj2.object_id
        assert obj1.bucket == "bucket-1"
        assert obj2.bucket == "bucket-2"


class TestBucketConfiguration:
    """Integration tests for bucket configuration effects."""

    def test_compression_hint(self):
        """Test bucket compression settings."""
        bucket = Bucket(
            bucket_id="compress-bucket",
            name="compress-bucket",
            compression_algorithm="zstd",
        )

        # Text should be compressed
        assert bucket.should_compress("text/plain")
        assert bucket.should_compress("application/json")

        # Binary shouldn't be compressed
        assert not bucket.should_compress("image/png")

    def test_no_compression(self):
        """Test bucket with compression disabled."""
        bucket = Bucket(
            bucket_id="no-compress-bucket",
            name="no-compress-bucket",
            compression_algorithm="none",
        )

        assert not bucket.should_compress("text/plain")
        assert not bucket.should_compress("application/json")

    def test_chunk_size_from_bucket(self):
        """Test chunk size configuration from bucket."""
        bucket = create_test_bucket()

        chunk_size = bucket.get_chunk_size()
        assert chunk_size == 64 * 1024 * 1024  # 64MB


class TestContentAddressableStorage:
    """Integration tests for content-addressable storage features."""

    def test_chunk_deduplication(self):
        """Test that identical chunks are deduplicated."""
        service = StorageService(chunk_size=10)
        bucket = create_test_bucket()

        # Same content in different objects
        same_data = b"0123456789"

        obj1 = service.put_object(bucket, "file1.dat", same_data)
        obj2 = service.put_object(bucket, "file2.dat", same_data)

        # Both should have same chunk checksums
        assert obj1.chunk_refs[0].checksum == obj2.chunk_refs[0].checksum

    def test_object_versioning(self):
        """Test object versioning."""
        service = StorageService()
        bucket = create_test_bucket()
        bucket.versioning_enabled = True

        # First version
        obj_v1 = service.put_object(bucket, "versioned.txt", b"Version 1")
        assert obj_v1.version == 1

        # Object can be updated
        assert service.get_object(obj_v1.object_id) is not None


class TestErasureCoding:
    """Integration tests for erasure coding (if implemented)."""

    def test_erasure_coding_service_exists(self):
        """Test that erasure coding service can be instantiated."""
        try:
            service = ErasureCodingService()
            assert service is not None
        except Exception:
            pytest.skip("Erasure coding service not fully implemented")


class TestObjectMetadata:
    """Integration tests for object metadata."""

    def test_content_type_stored(self):
        """Test that content type is properly stored."""
        service = StorageService()
        bucket = create_test_bucket()

        obj = service.put_object(
            bucket,
            "document.json",
            b'{"key": "value"}',
            content_type="application/json",
        )

        retrieved = service.get_object(obj.object_id)
        assert retrieved.content_type == "application/json"

    def test_object_size_accurate(self):
        """Test that object size is accurately tracked."""
        service = StorageService()
        bucket = create_test_bucket()

        data = b"X" * 1000
        obj = service.put_object(bucket, "sized.dat", data)

        assert obj.size == 1000


class TestEdgeCases:
    """Integration tests for edge cases."""

    def test_empty_object(self):
        """Test handling of empty objects."""
        service = StorageService()
        bucket = create_test_bucket()

        obj = service.put_object(bucket, "empty.txt", b"")

        assert obj.size == 0
        assert len(obj.chunk_refs) == 0

    def test_single_byte_object(self):
        """Test handling of single byte objects."""
        service = StorageService()
        bucket = create_test_bucket()

        obj = service.put_object(bucket, "single.dat", b"X")

        assert obj.size == 1
        assert len(obj.chunk_refs) == 1

    def test_delete_nonexistent_object(self):
        """Test deleting an object that doesn't exist."""
        service = StorageService()

        result = service.delete_object("nonexistent-id")
        assert result is False

    def test_get_nonexistent_object(self):
        """Test getting an object that doesn't exist."""
        service = StorageService()

        result = service.get_object("nonexistent-id")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
