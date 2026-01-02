"""Unit tests for object_store domain layer."""

import pytest
from datetime import datetime

from object_store.domain.entities.bucket import Bucket
from object_store.domain.entities.chunk import Chunk, ChunkRef
from object_store.domain.entities.object import Object, ObjectVersion
from object_store.domain.services.storage_service import StorageService
from object_store.domain.services.erasure_coding_service import ErasureCodingService


@pytest.mark.unit
class TestBucket:
    """Test bucket entity."""

    def test_bucket_creation(self):
        """Test creating a bucket."""
        bucket = Bucket(
            bucket_id="bucket-001",
            name="my-bucket",
        )
        assert bucket.name == "my-bucket"
        assert bucket.versioning_enabled is True
        assert bucket.compression_algorithm == "zstd"

    def test_should_compress_text(self):
        """Test compression decision for text content."""
        bucket = Bucket(bucket_id="b1", name="test")
        assert bucket.should_compress("text/plain")
        assert bucket.should_compress("text/html")
        assert bucket.should_compress("application/json")

    def test_should_not_compress_binary(self):
        """Test compression decision for binary content."""
        bucket = Bucket(bucket_id="b1", name="test")
        assert not bucket.should_compress("image/png")
        assert not bucket.should_compress("application/octet-stream")

    def test_compression_disabled(self):
        """Test when compression is disabled."""
        bucket = Bucket(
            bucket_id="b1",
            name="test",
            compression_algorithm="none",
        )
        assert not bucket.should_compress("text/plain")

    def test_get_chunk_size(self):
        """Test chunk size recommendation."""
        bucket = Bucket(bucket_id="b1", name="test")
        assert bucket.get_chunk_size() == 64 * 1024 * 1024


@pytest.mark.unit
class TestChunk:
    """Test chunk entity."""

    def test_chunk_creation(self):
        """Test creating a chunk."""
        data = b"Hello, World!"
        chunk = Chunk(
            chunk_id="chunk-001",
            object_id="obj-001",
            sequence=0,
            data=data,
            size=len(data),
            checksum="",
        )
        assert chunk.size == 13

    def test_calculate_checksum(self):
        """Test checksum calculation."""
        data = b"Hello, World!"
        chunk = Chunk(
            chunk_id="chunk-001",
            object_id="obj-001",
            sequence=0,
            data=data,
            size=len(data),
            checksum="",
        )
        checksum = chunk.calculate_checksum()
        assert len(checksum) == 64  # SHA256 hex

    def test_verify_checksum_valid(self):
        """Test checksum verification with valid data."""
        data = b"Hello, World!"
        chunk = Chunk(
            chunk_id="chunk-001",
            object_id="obj-001",
            sequence=0,
            data=data,
            size=len(data),
            checksum="",
        )
        chunk.checksum = chunk.calculate_checksum()
        assert chunk.verify_checksum()

    def test_verify_checksum_invalid(self):
        """Test checksum verification with corrupted data."""
        data = b"Hello, World!"
        chunk = Chunk(
            chunk_id="chunk-001",
            object_id="obj-001",
            sequence=0,
            data=data,
            size=len(data),
            checksum="invalid_checksum",
        )
        assert not chunk.verify_checksum()


@pytest.mark.unit
class TestChunkRef:
    """Test chunk reference."""

    def test_chunk_ref_creation(self):
        """Test creating a chunk reference."""
        ref = ChunkRef(
            chunk_id="chunk-001",
            checksum="abc123",
            sequence=0,
            size=1024,
            shard_indices=[0, 1, 2],
        )
        assert ref.chunk_id == "chunk-001"
        assert ref.shard_indices == [0, 1, 2]


@pytest.mark.unit
class TestObject:
    """Test object entity."""

    def test_object_creation(self):
        """Test creating an object."""
        obj = Object(
            object_id="obj-001",
            bucket="my-bucket",
            key="path/to/file.txt",
        )
        assert obj.version == 1
        assert obj.size == 0
        assert obj.content_type == "application/octet-stream"

    def test_get_full_path(self):
        """Test full path generation."""
        obj = Object(
            object_id="obj-001",
            bucket="my-bucket",
            key="path/to/file.txt",
        )
        assert obj.get_full_path() == "my-bucket/path/to/file.txt"

    def test_is_latest_version(self):
        """Test version checking."""
        obj = Object(
            object_id="obj-001",
            bucket="my-bucket",
            key="file.txt",
            version=1,
        )
        assert obj.is_latest_version()


@pytest.mark.unit
class TestObjectVersion:
    """Test object version tracking."""

    def test_object_version_creation(self):
        """Test creating an object version."""
        ver = ObjectVersion(
            object_id="obj-001",
            version=2,
            size=1024,
            checksum="abc123",
        )
        assert ver.version == 2
        assert ver.is_deleted is False

    def test_deleted_version(self):
        """Test deleted version marker."""
        ver = ObjectVersion(
            object_id="obj-001",
            version=3,
            is_deleted=True,
        )
        assert ver.is_deleted


@pytest.mark.unit
class TestStorageService:
    """Test storage service."""

    def test_put_object(self):
        """Test storing an object."""
        service = StorageService(chunk_size=1024)
        bucket = Bucket(bucket_id="b1", name="test-bucket")
        data = b"Hello, World! This is test data."

        obj = service.put_object(bucket, "test.txt", data, "text/plain")

        assert obj.bucket == "test-bucket"
        assert obj.key == "test.txt"
        assert obj.size == len(data)
        assert obj.content_type == "text/plain"

    def test_put_object_creates_chunks(self):
        """Test that put_object creates chunks."""
        service = StorageService()
        bucket = Bucket(bucket_id="b1", name="test-bucket")
        data = b"This is a longer piece of data that will be split into chunks"

        obj = service.put_object(bucket, "chunked.txt", data)

        # At least one chunk is created
        assert len(obj.chunk_refs) >= 1
        # Total chunk size equals data size
        total_chunk_size = sum(ref.size for ref in obj.chunk_refs)
        assert total_chunk_size == len(data)
        # All chunks have valid checksums
        for ref in obj.chunk_refs:
            assert len(ref.checksum) == 64  # SHA256 hex

    def test_get_object(self):
        """Test retrieving an object."""
        service = StorageService()
        bucket = Bucket(bucket_id="b1", name="test-bucket")
        data = b"Hello, World!"

        obj = service.put_object(bucket, "test.txt", data)
        retrieved = service.get_object(obj.object_id)

        assert retrieved is not None
        assert retrieved.object_id == obj.object_id

    def test_get_nonexistent_object(self):
        """Test retrieving a nonexistent object."""
        service = StorageService()
        assert service.get_object("nonexistent") is None

    def test_delete_object(self):
        """Test deleting an object."""
        service = StorageService()
        bucket = Bucket(bucket_id="b1", name="test-bucket")
        data = b"Hello, World!"

        obj = service.put_object(bucket, "test.txt", data)
        result = service.delete_object(obj.object_id)

        assert result is True
        assert service.get_object(obj.object_id) is None

    def test_delete_nonexistent_object(self):
        """Test deleting a nonexistent object."""
        service = StorageService()
        assert service.delete_object("nonexistent") is False

    def test_verify_object_integrity(self):
        """Test object integrity verification."""
        service = StorageService()
        bucket = Bucket(bucket_id="b1", name="test-bucket")
        data = b"Hello, World!"

        obj = service.put_object(bucket, "test.txt", data)
        assert service.verify_object_integrity(obj.object_id) is True

    def test_verify_integrity_nonexistent(self):
        """Test integrity check for nonexistent object."""
        service = StorageService()
        assert service.verify_object_integrity("nonexistent") is False


@pytest.mark.unit
class TestErasureCodingService:
    """Test erasure coding service."""

    def test_encode_creates_correct_shard_count(self):
        """Test that encoding creates correct number of shards."""
        ec = ErasureCodingService(data_shards=4, parity_shards=2)
        data = b"Hello, World! This is test data for erasure coding."

        shards = ec.encode(data)

        assert len(shards) == 6  # 4 data + 2 parity

    def test_decode_with_all_shards(self):
        """Test decoding with all shards present."""
        ec = ErasureCodingService(data_shards=4, parity_shards=2)
        original = b"Hello, World! This is test data."

        shards = ec.encode(original)
        decoded = ec.decode(shards)

        assert decoded == original

    def test_decode_with_missing_data_shard(self):
        """Test decoding with one missing data shard."""
        ec = ErasureCodingService(data_shards=4, parity_shards=2)
        original = b"Hello, World! This is test data."

        shards = ec.encode(original)
        # Remove one data shard
        shards[1] = None

        decoded = ec.decode(shards)

        assert decoded == original

    def test_decode_with_missing_parity_shard(self):
        """Test decoding with missing parity shard."""
        ec = ErasureCodingService(data_shards=4, parity_shards=2)
        original = b"Hello, World! This is test data."

        shards = ec.encode(original)
        # Remove parity shard
        shards[4] = None

        decoded = ec.decode(shards)

        assert decoded == original

    def test_decode_unrecoverable(self):
        """Test decoding when too many shards are missing."""
        ec = ErasureCodingService(data_shards=4, parity_shards=2)
        original = b"Hello, World! This is test data."

        shards = ec.encode(original)
        # Remove too many shards
        shards[0] = None
        shards[1] = None
        shards[2] = None

        decoded = ec.decode(shards)

        assert decoded is None

    def test_can_recover_sufficient_shards(self):
        """Test can_recover with sufficient shards."""
        ec = ErasureCodingService(data_shards=4, parity_shards=2)
        assert ec.can_recover(4) is True
        assert ec.can_recover(5) is True
        assert ec.can_recover(6) is True

    def test_can_recover_insufficient_shards(self):
        """Test can_recover with insufficient shards."""
        ec = ErasureCodingService(data_shards=4, parity_shards=2)
        assert ec.can_recover(3) is False
        assert ec.can_recover(0) is False

    def test_min_shards_required(self):
        """Test minimum shards required."""
        ec = ErasureCodingService(data_shards=4, parity_shards=2)
        assert ec.min_shards_required() == 4

    def test_encode_decode_roundtrip(self):
        """Test full encode-decode roundtrip."""
        ec = ErasureCodingService(data_shards=4, parity_shards=2)
        original = b"Test data for erasure coding roundtrip"

        shards = ec.encode(original)
        decoded = ec.decode(shards)

        assert decoded == original

    def test_encode_empty_data(self):
        """Test encoding empty data."""
        ec = ErasureCodingService(data_shards=4, parity_shards=2)
        shards = ec.encode(b"")

        assert len(shards) == 6
        # All shards should be empty/zero
        for shard in shards:
            assert shard == b"\x00" or len(shard) == 0 or all(b == 0 for b in shard)

    def test_shard_sizes_consistent(self):
        """Test that all shards have consistent size."""
        ec = ErasureCodingService(data_shards=4, parity_shards=2)
        data = b"Hello, World! This is some test data that will be encoded."

        shards = ec.encode(data)

        sizes = [len(s) for s in shards]
        assert len(set(sizes)) == 1  # All same size
