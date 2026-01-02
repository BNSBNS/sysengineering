"""FastAPI REST adapter for the Streaming System.

Provides HTTP endpoints for message streaming with Raft consensus.

Usage:
    from streaming_system.adapters.inbound.rest_api import create_app

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
    from fastapi import FastAPI, HTTPException, status
    from pydantic import BaseModel, Field

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

from streaming_system.domain.entities.record import Record
from streaming_system.domain.services.partition_log import PartitionLog
from streaming_system.domain.services.consumer_coordinator import ConsumerCoordinator


# Pydantic models for request/response serialization
if _HAS_FASTAPI:

    class ProduceRecordRequest(BaseModel):
        """Request to produce a record."""

        key: str = Field(..., description="Record key (base64)")
        value: str = Field(..., description="Record value (base64)")
        headers: dict = Field(default_factory=dict, description="Record headers")

    class ProduceResponse(BaseModel):
        """Produce result."""

        partition_id: int
        offset: int

    class BatchProduceRequest(BaseModel):
        """Request to produce multiple records."""

        records: list[ProduceRecordRequest]

    class BatchProduceResponse(BaseModel):
        """Batch produce result."""

        partition_id: int
        offsets: list[int]
        count: int

    class RecordResponse(BaseModel):
        """Record details."""

        offset: int
        term: int
        key: str
        value: str
        timestamp: float
        is_committed: bool
        headers: dict

    class FetchResponse(BaseModel):
        """Fetch result."""

        partition_id: int
        records: list[RecordResponse]
        next_offset: int
        hwm: int

    class CreateGroupRequest(BaseModel):
        """Request to create consumer group."""

        group_id: str = Field(..., min_length=1, description="Group ID")
        protocol_type: str = Field(default="range", description="Assignment: range, roundrobin")

    class JoinGroupRequest(BaseModel):
        """Request to join consumer group."""

        consumer_id: str = Field(..., min_length=1, description="Consumer ID")
        session_timeout_ms: int = Field(default=10000, ge=1000, description="Session timeout")

    class GroupResponse(BaseModel):
        """Consumer group details."""

        group_id: str
        state: str
        protocol_type: str
        generation: int
        member_count: int
        members: list[str]

    class CommitOffsetRequest(BaseModel):
        """Request to commit offset."""

        partition_id: int = Field(..., ge=0, description="Partition ID")
        offset: int = Field(..., ge=0, description="Offset to commit")

    class OffsetResponse(BaseModel):
        """Offset details."""

        group_id: str
        partition_id: int
        offset: int

    class PartitionAssignmentResponse(BaseModel):
        """Partition assignment result."""

        group_id: str
        assignments: dict

    class PartitionStatsResponse(BaseModel):
        """Partition statistics."""

        partition_id: int
        total_records: int
        committed_offset: int
        segments: int
        current_term: int

    class MetricsResponse(BaseModel):
        """System metrics."""

        partitions: int
        total_records: int
        consumer_groups: int

    class HealthResponse(BaseModel):
        """Health check response."""

        status: str
        version: str = "0.1.0"


def create_app(
    partition_count: int = 4,
) -> "FastAPI":
    """Create FastAPI application with Streaming System endpoints.

    Args:
        partition_count: Number of partitions to create.

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
    partitions: dict[int, PartitionLog] = {}
    for i in range(partition_count):
        partitions[i] = PartitionLog(partition_id=i)

    coordinator = ConsumerCoordinator()

    app = FastAPI(
        title="Streaming System API",
        description="Distributed message streaming with Raft consensus and consumer groups",
        version="1.0.0",
    )

    def _get_partition(key: bytes) -> int:
        """Get partition for key using consistent hashing."""
        return hash(key) % len(partitions)

    # Health endpoints
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Check system health status."""
        return HealthResponse(status="healthy")

    @app.get("/metrics", response_model=MetricsResponse, tags=["System"])
    async def get_metrics():
        """Get system metrics."""
        total_records = sum(p.get_next_offset() for p in partitions.values())
        return MetricsResponse(
            partitions=len(partitions),
            total_records=total_records,
            consumer_groups=len(coordinator._groups),
        )

    # Producer endpoints
    @app.post(
        "/produce/{partition_id}",
        response_model=ProduceResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Producer"],
    )
    async def produce_to_partition(partition_id: int, request: ProduceRecordRequest):
        """Produce a record to a specific partition."""
        if partition_id not in partitions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Partition {partition_id} not found",
            )

        try:
            key = base64.b64decode(request.key)
            value = base64.b64decode(request.value)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid base64 encoding",
            )

        record = Record(key=key, value=value, headers=request.headers)
        offsets = partitions[partition_id].append([record])

        return ProduceResponse(partition_id=partition_id, offset=offsets[0])

    @app.post(
        "/produce",
        response_model=ProduceResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Producer"],
    )
    async def produce(request: ProduceRecordRequest):
        """Produce a record (auto-partition by key)."""
        try:
            key = base64.b64decode(request.key)
            value = base64.b64decode(request.value)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid base64 encoding",
            )

        partition_id = _get_partition(key)
        record = Record(key=key, value=value, headers=request.headers)
        offsets = partitions[partition_id].append([record])

        return ProduceResponse(partition_id=partition_id, offset=offsets[0])

    @app.post(
        "/produce/{partition_id}/batch",
        response_model=BatchProduceResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Producer"],
    )
    async def batch_produce(partition_id: int, request: BatchProduceRequest):
        """Produce multiple records to a partition."""
        if partition_id not in partitions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Partition {partition_id} not found",
            )

        records = []
        for r in request.records:
            try:
                key = base64.b64decode(r.key)
                value = base64.b64decode(r.value)
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid base64 encoding",
                )
            records.append(Record(key=key, value=value, headers=r.headers))

        offsets = partitions[partition_id].append(records)

        return BatchProduceResponse(
            partition_id=partition_id,
            offsets=offsets,
            count=len(offsets),
        )

    # Consumer endpoints
    @app.get(
        "/fetch/{partition_id}",
        response_model=FetchResponse,
        tags=["Consumer"],
    )
    async def fetch(partition_id: int, offset: int = 0, max_records: int = 100):
        """Fetch records from a partition."""
        if partition_id not in partitions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Partition {partition_id} not found",
            )

        partition = partitions[partition_id]
        entries = partition.read(offset, max_records)

        records = [
            RecordResponse(
                offset=e.offset,
                term=e.term,
                key=base64.b64encode(e.record.key).decode(),
                value=base64.b64encode(e.record.value).decode(),
                timestamp=e.record.timestamp,
                is_committed=e.is_committed,
                headers=e.record.headers,
            )
            for e in entries
        ]

        next_offset = entries[-1].offset + 1 if entries else offset

        return FetchResponse(
            partition_id=partition_id,
            records=records,
            next_offset=next_offset,
            hwm=partition.get_hwm(),
        )

    @app.post("/partitions/{partition_id}/commit", tags=["Consumer"])
    async def commit_partition(partition_id: int, offset: int):
        """Commit offset on a partition (advance HWM)."""
        if partition_id not in partitions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Partition {partition_id} not found",
            )

        partitions[partition_id].commit(offset)
        return {"partition_id": partition_id, "committed_offset": offset}

    # Consumer group endpoints
    @app.post(
        "/groups",
        response_model=GroupResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Consumer Groups"],
    )
    async def create_group(request: CreateGroupRequest):
        """Create a consumer group."""
        if coordinator.get_group(request.group_id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Group {request.group_id} already exists",
            )

        group = coordinator.create_group(request.group_id, request.protocol_type)
        return GroupResponse(
            group_id=group.group_id,
            state=group.state.value,
            protocol_type=group.protocol_type,
            generation=group.generation,
            member_count=group.size(),
            members=list(group.members.keys()),
        )

    @app.get("/groups", tags=["Consumer Groups"])
    async def list_groups():
        """List all consumer groups."""
        groups = [
            GroupResponse(
                group_id=g.group_id,
                state=g.state.value,
                protocol_type=g.protocol_type,
                generation=g.generation,
                member_count=g.size(),
                members=list(g.members.keys()),
            )
            for g in coordinator._groups.values()
        ]
        return {"groups": groups, "count": len(groups)}

    @app.get("/groups/{group_id}", response_model=GroupResponse, tags=["Consumer Groups"])
    async def get_group(group_id: str):
        """Get consumer group details."""
        group = coordinator.get_group(group_id)
        if not group:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Group {group_id} not found",
            )
        return GroupResponse(
            group_id=group.group_id,
            state=group.state.value,
            protocol_type=group.protocol_type,
            generation=group.generation,
            member_count=group.size(),
            members=list(group.members.keys()),
        )

    @app.post("/groups/{group_id}/join", response_model=GroupResponse, tags=["Consumer Groups"])
    async def join_group(group_id: str, request: JoinGroupRequest):
        """Join a consumer to a group."""
        group = coordinator.join_group(
            group_id=group_id,
            consumer_id=request.consumer_id,
            session_timeout_ms=request.session_timeout_ms,
        )
        return GroupResponse(
            group_id=group.group_id,
            state=group.state.value,
            protocol_type=group.protocol_type,
            generation=group.generation,
            member_count=group.size(),
            members=list(group.members.keys()),
        )

    @app.post("/groups/{group_id}/leave", tags=["Consumer Groups"])
    async def leave_group(group_id: str, consumer_id: str):
        """Remove a consumer from a group."""
        group = coordinator.leave_group(group_id, consumer_id)
        if group:
            return {"status": "left", "group_id": group_id, "consumer_id": consumer_id}
        return {"status": "group_empty", "group_id": group_id}

    @app.post("/groups/{group_id}/heartbeat", tags=["Consumer Groups"])
    async def heartbeat(group_id: str, consumer_id: str):
        """Send consumer heartbeat."""
        result = coordinator.heartbeat(group_id, consumer_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Consumer {consumer_id} not in group {group_id}",
            )
        return {"status": "ok", "group_id": group_id, "consumer_id": consumer_id}

    @app.post("/groups/{group_id}/offsets", response_model=OffsetResponse, tags=["Consumer Groups"])
    async def commit_offset(group_id: str, request: CommitOffsetRequest):
        """Commit consumer offset."""
        coordinator.commit_offset(group_id, request.partition_id, request.offset)
        return OffsetResponse(
            group_id=group_id,
            partition_id=request.partition_id,
            offset=request.offset,
        )

    @app.get("/groups/{group_id}/offsets/{partition_id}", response_model=OffsetResponse, tags=["Consumer Groups"])
    async def get_offset(group_id: str, partition_id: int):
        """Get committed offset for partition."""
        offset = coordinator.get_offset(group_id, partition_id)
        return OffsetResponse(
            group_id=group_id,
            partition_id=partition_id,
            offset=offset,
        )

    @app.post("/groups/{group_id}/assign", response_model=PartitionAssignmentResponse, tags=["Consumer Groups"])
    async def assign_partitions(group_id: str):
        """Assign partitions to consumers in group."""
        assignments = coordinator.assign_partitions(group_id, len(partitions))
        return PartitionAssignmentResponse(
            group_id=group_id,
            assignments=assignments,
        )

    # Partition endpoints
    @app.get("/partitions", tags=["Partitions"])
    async def list_partitions():
        """List all partitions."""
        stats = [p.get_stats() for p in partitions.values()]
        return {"partitions": stats, "count": len(stats)}

    @app.get("/partitions/{partition_id}", response_model=PartitionStatsResponse, tags=["Partitions"])
    async def get_partition_stats(partition_id: int):
        """Get partition statistics."""
        if partition_id not in partitions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Partition {partition_id} not found",
            )
        stats = partitions[partition_id].get_stats()
        return PartitionStatsResponse(**stats)

    return app
