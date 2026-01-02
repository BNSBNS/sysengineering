"""FastAPI REST adapter for the Object Store.

Provides HTTP endpoints for object storage with versioning and erasure coding.

Usage:
    from object_store.adapters.inbound.rest_api import create_app

    app = create_app()
    # Run with: uvicorn module:app --host 0.0.0.0 --port 8080

References:
    - design.md Section 2.4 (API Services)
    - ports/inbound/api.py (API contracts)
"""

from __future__ import annotations

import base64
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form
    from fastapi.responses import Response
    from pydantic import BaseModel, Field

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

from object_store.domain.entities.bucket import Bucket
from object_store.domain.entities.object import Object
from object_store.domain.services.storage_service import StorageService
from object_store.domain.services.erasure_coding_service import ErasureCodingService


# Pydantic models for request/response serialization
if _HAS_FASTAPI:

    class CreateBucketRequest(BaseModel):
        """Request to create a bucket."""

        name: str = Field(..., min_length=3, max_length=63, description="Bucket name")
        versioning_enabled: bool = Field(default=True, description="Enable versioning")
        deduplication_enabled: bool = Field(default=True, description="Enable deduplication")
        compression_algorithm: str = Field(default="zstd", description="Compression: zstd, gzip, none")
        replication_factor: int = Field(default=3, ge=1, le=5, description="Replication factor")
        erasure_parity_shards: int = Field(default=2, ge=0, description="Erasure coding parity shards")

    class BucketResponse(BaseModel):
        """Bucket details response."""

        bucket_id: str
        name: str
        versioning_enabled: bool
        deduplication_enabled: bool
        compression_algorithm: str
        replication_factor: int
        erasure_parity_shards: int
        object_count: int
        total_size: int

    class BucketListResponse(BaseModel):
        """List of buckets."""

        buckets: list[BucketResponse]
        count: int

    class PutObjectRequest(BaseModel):
        """Request to put an object (JSON mode)."""

        key: str = Field(..., min_length=1, description="Object key/path")
        data_base64: str = Field(..., description="Base64-encoded object data")
        content_type: str = Field(default="application/octet-stream", description="MIME type")
        metadata: dict = Field(default_factory=dict, description="Custom metadata")

    class ObjectResponse(BaseModel):
        """Object details response."""

        object_id: str
        bucket: str
        key: str
        version: int
        size: int
        checksum: str
        content_type: str
        metadata: dict
        chunk_count: int
        replication_factor: int
        erasure_parity_shards: int

    class ObjectListResponse(BaseModel):
        """List of objects."""

        objects: list[ObjectResponse]
        count: int

    class ChunkRefResponse(BaseModel):
        """Chunk reference details."""

        chunk_id: str
        checksum: str
        sequence: int
        size: int
        shard_indices: list[int]

    class ObjectDetailResponse(BaseModel):
        """Detailed object response with chunks."""

        object_id: str
        bucket: str
        key: str
        version: int
        size: int
        checksum: str
        content_type: str
        metadata: dict
        chunk_refs: list[ChunkRefResponse]

    class IntegrityCheckResponse(BaseModel):
        """Integrity check result."""

        object_id: str
        valid: bool
        chunks_checked: int

    class ErasureCodeRequest(BaseModel):
        """Request to encode data with erasure coding."""

        data_base64: str = Field(..., description="Base64-encoded data")
        data_shards: int = Field(default=4, ge=2, description="Number of data shards")
        parity_shards: int = Field(default=2, ge=1, description="Number of parity shards")

    class ErasureCodeResponse(BaseModel):
        """Erasure coding result."""

        shards_base64: list[str]
        data_shards: int
        parity_shards: int
        total_shards: int

    class ErasureDecodeRequest(BaseModel):
        """Request to decode erasure coded data."""

        shards_base64: list[Optional[str]] = Field(..., description="Shards (null for missing)")
        data_shards: int = Field(default=4, ge=2, description="Number of data shards")
        parity_shards: int = Field(default=2, ge=1, description="Number of parity shards")

    class ErasureDecodeResponse(BaseModel):
        """Erasure decoding result."""

        data_base64: Optional[str]
        success: bool
        shards_available: int
        shards_required: int

    class MetricsResponse(BaseModel):
        """Store metrics."""

        buckets_count: int
        objects_count: int
        chunks_count: int
        total_size_bytes: int

    class HealthResponse(BaseModel):
        """Health check response."""

        status: str
        version: str = "0.1.0"


def create_app(
    storage_service: StorageService | None = None,
) -> "FastAPI":
    """Create FastAPI application with Object Store endpoints.

    Args:
        storage_service: Optional StorageService instance.

    Returns:
        Configured FastAPI application.

    Raises:
        ImportError: If FastAPI is not installed.
    """
    if not _HAS_FASTAPI:
        raise ImportError(
            "FastAPI not installed. Install with: pip install fastapi uvicorn"
        )

    # Initialize services
    storage = storage_service or StorageService()
    buckets: dict[str, Bucket] = {}
    next_bucket_id = 0

    app = FastAPI(
        title="Object Store API",
        description="Versioned object storage with erasure coding and deduplication",
        version="1.0.0",
    )

    # Health endpoints
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Check store health status."""
        return HealthResponse(status="healthy")

    @app.get("/metrics", response_model=MetricsResponse, tags=["System"])
    async def get_metrics():
        """Get store metrics."""
        total_size = sum(obj.size for obj in storage.objects.values())
        return MetricsResponse(
            buckets_count=len(buckets),
            objects_count=len(storage.objects),
            chunks_count=len(storage.chunks),
            total_size_bytes=total_size,
        )

    # Bucket endpoints
    @app.post(
        "/buckets",
        response_model=BucketResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Buckets"],
    )
    async def create_bucket(request: CreateBucketRequest):
        """Create a new bucket."""
        nonlocal next_bucket_id

        if request.name in [b.name for b in buckets.values()]:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Bucket {request.name} already exists",
            )

        bucket_id = f"bucket-{next_bucket_id}"
        next_bucket_id += 1

        bucket = Bucket(
            bucket_id=bucket_id,
            name=request.name,
            versioning_enabled=request.versioning_enabled,
            deduplication_enabled=request.deduplication_enabled,
            compression_algorithm=request.compression_algorithm,
            replication_factor=request.replication_factor,
            erasure_parity_shards=request.erasure_parity_shards,
        )
        buckets[bucket_id] = bucket

        return BucketResponse(
            bucket_id=bucket.bucket_id,
            name=bucket.name,
            versioning_enabled=bucket.versioning_enabled,
            deduplication_enabled=bucket.deduplication_enabled,
            compression_algorithm=bucket.compression_algorithm,
            replication_factor=bucket.replication_factor,
            erasure_parity_shards=bucket.erasure_parity_shards,
            object_count=bucket.object_count,
            total_size=bucket.total_size,
        )

    @app.get("/buckets", response_model=BucketListResponse, tags=["Buckets"])
    async def list_buckets():
        """List all buckets."""
        return BucketListResponse(
            buckets=[
                BucketResponse(
                    bucket_id=b.bucket_id,
                    name=b.name,
                    versioning_enabled=b.versioning_enabled,
                    deduplication_enabled=b.deduplication_enabled,
                    compression_algorithm=b.compression_algorithm,
                    replication_factor=b.replication_factor,
                    erasure_parity_shards=b.erasure_parity_shards,
                    object_count=b.object_count,
                    total_size=b.total_size,
                )
                for b in buckets.values()
            ],
            count=len(buckets),
        )

    @app.get("/buckets/{bucket_name}", response_model=BucketResponse, tags=["Buckets"])
    async def get_bucket(bucket_name: str):
        """Get bucket details."""
        bucket = next((b for b in buckets.values() if b.name == bucket_name), None)
        if not bucket:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bucket {bucket_name} not found",
            )
        return BucketResponse(
            bucket_id=bucket.bucket_id,
            name=bucket.name,
            versioning_enabled=bucket.versioning_enabled,
            deduplication_enabled=bucket.deduplication_enabled,
            compression_algorithm=bucket.compression_algorithm,
            replication_factor=bucket.replication_factor,
            erasure_parity_shards=bucket.erasure_parity_shards,
            object_count=bucket.object_count,
            total_size=bucket.total_size,
        )

    @app.delete("/buckets/{bucket_name}", status_code=status.HTTP_204_NO_CONTENT, tags=["Buckets"])
    async def delete_bucket(bucket_name: str):
        """Delete a bucket."""
        bucket = next((b for b in buckets.values() if b.name == bucket_name), None)
        if not bucket:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bucket {bucket_name} not found",
            )

        # Check if bucket has objects
        bucket_objects = [o for o in storage.objects.values() if o.bucket == bucket_name]
        if bucket_objects:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Bucket {bucket_name} is not empty",
            )

        del buckets[bucket.bucket_id]

    # Object endpoints
    @app.post(
        "/buckets/{bucket_name}/objects",
        response_model=ObjectResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Objects"],
    )
    async def put_object(bucket_name: str, request: PutObjectRequest):
        """Put an object in a bucket (JSON mode with base64 data)."""
        bucket = next((b for b in buckets.values() if b.name == bucket_name), None)
        if not bucket:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bucket {bucket_name} not found",
            )

        try:
            data = base64.b64decode(request.data_base64)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid base64 data",
            )

        obj = storage.put_object(
            bucket=bucket,
            key=request.key,
            data=data,
            content_type=request.content_type,
        )
        obj.metadata = request.metadata

        # Update bucket stats
        bucket.object_count += 1
        bucket.total_size += obj.size

        return ObjectResponse(
            object_id=obj.object_id,
            bucket=obj.bucket,
            key=obj.key,
            version=obj.version,
            size=obj.size,
            checksum=obj.checksum,
            content_type=obj.content_type,
            metadata=obj.metadata,
            chunk_count=len(obj.chunk_refs),
            replication_factor=obj.replication_factor,
            erasure_parity_shards=obj.erasure_parity_shards,
        )

    @app.get("/buckets/{bucket_name}/objects", response_model=ObjectListResponse, tags=["Objects"])
    async def list_objects(bucket_name: str):
        """List objects in a bucket."""
        bucket = next((b for b in buckets.values() if b.name == bucket_name), None)
        if not bucket:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bucket {bucket_name} not found",
            )

        bucket_objects = [o for o in storage.objects.values() if o.bucket == bucket_name]
        return ObjectListResponse(
            objects=[
                ObjectResponse(
                    object_id=o.object_id,
                    bucket=o.bucket,
                    key=o.key,
                    version=o.version,
                    size=o.size,
                    checksum=o.checksum,
                    content_type=o.content_type,
                    metadata=o.metadata,
                    chunk_count=len(o.chunk_refs),
                    replication_factor=o.replication_factor,
                    erasure_parity_shards=o.erasure_parity_shards,
                )
                for o in bucket_objects
            ],
            count=len(bucket_objects),
        )

    @app.get("/objects/{object_id}", response_model=ObjectDetailResponse, tags=["Objects"])
    async def get_object(object_id: str):
        """Get object details."""
        obj = storage.get_object(object_id)
        if not obj:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Object {object_id} not found",
            )
        return ObjectDetailResponse(
            object_id=obj.object_id,
            bucket=obj.bucket,
            key=obj.key,
            version=obj.version,
            size=obj.size,
            checksum=obj.checksum,
            content_type=obj.content_type,
            metadata=obj.metadata,
            chunk_refs=[
                ChunkRefResponse(
                    chunk_id=ref.chunk_id,
                    checksum=ref.checksum,
                    sequence=ref.sequence,
                    size=ref.size,
                    shard_indices=ref.shard_indices,
                )
                for ref in obj.chunk_refs
            ],
        )

    @app.get("/objects/{object_id}/data", tags=["Objects"])
    async def get_object_data(object_id: str):
        """Get object data (reconstructed from chunks)."""
        obj = storage.get_object(object_id)
        if not obj:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Object {object_id} not found",
            )

        # Reconstruct data from chunks
        data = b""
        for ref in sorted(obj.chunk_refs, key=lambda r: r.sequence):
            chunk = storage.chunks.get(ref.chunk_id)
            if not chunk:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Chunk {ref.chunk_id} not found",
                )
            data += chunk.data

        return Response(
            content=data,
            media_type=obj.content_type,
            headers={"X-Object-Checksum": obj.checksum},
        )

    @app.delete("/objects/{object_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Objects"])
    async def delete_object(object_id: str):
        """Delete an object."""
        obj = storage.get_object(object_id)
        if not obj:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Object {object_id} not found",
            )

        # Update bucket stats
        bucket = next((b for b in buckets.values() if b.name == obj.bucket), None)
        if bucket:
            bucket.object_count = max(0, bucket.object_count - 1)
            bucket.total_size = max(0, bucket.total_size - obj.size)

        storage.delete_object(object_id)

    @app.get("/objects/{object_id}/verify", response_model=IntegrityCheckResponse, tags=["Objects"])
    async def verify_object_integrity(object_id: str):
        """Verify object integrity via chunk checksums."""
        obj = storage.get_object(object_id)
        if not obj:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Object {object_id} not found",
            )

        valid = storage.verify_object_integrity(object_id)
        return IntegrityCheckResponse(
            object_id=object_id,
            valid=valid,
            chunks_checked=len(obj.chunk_refs),
        )

    # Erasure coding endpoints
    @app.post("/erasure/encode", response_model=ErasureCodeResponse, tags=["Erasure Coding"])
    async def erasure_encode(request: ErasureCodeRequest):
        """Encode data with erasure coding."""
        try:
            data = base64.b64decode(request.data_base64)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid base64 data",
            )

        ec = ErasureCodingService(
            data_shards=request.data_shards,
            parity_shards=request.parity_shards,
        )
        shards = ec.encode(data)

        return ErasureCodeResponse(
            shards_base64=[base64.b64encode(s).decode() for s in shards],
            data_shards=request.data_shards,
            parity_shards=request.parity_shards,
            total_shards=len(shards),
        )

    @app.post("/erasure/decode", response_model=ErasureDecodeResponse, tags=["Erasure Coding"])
    async def erasure_decode(request: ErasureDecodeRequest):
        """Decode data from erasure coded shards."""
        shards = []
        for shard_b64 in request.shards_base64:
            if shard_b64 is None:
                shards.append(None)
            else:
                try:
                    shards.append(base64.b64decode(shard_b64))
                except Exception:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid base64 shard data",
                    )

        ec = ErasureCodingService(
            data_shards=request.data_shards,
            parity_shards=request.parity_shards,
        )

        available = sum(1 for s in shards if s is not None)
        data = ec.decode(shards)

        return ErasureDecodeResponse(
            data_base64=base64.b64encode(data).decode() if data else None,
            success=data is not None,
            shards_available=available,
            shards_required=ec.min_shards_required(),
        )

    return app
