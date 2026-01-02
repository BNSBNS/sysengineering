"""Ports layer - interface definitions following Hexagonal Architecture.

Ports are abstract interfaces (protocols) that define contracts:
- Inbound ports: APIs offered to clients (e.g., BufferPool, TransactionManager)
- Outbound ports: Dependencies on external systems (e.g., DiskManager, WALWriter)

Adapters implement these ports with concrete functionality.
"""

from db_engine.ports.inbound import (
    BufferPool,
    BufferPoolFullError,
    BufferPoolStats,
    CheckpointData,
    Index,
    IndexKey,
    IndexManager,
    IndexMetadata,
    IndexStats,
    KeyType,
    LockConflictError,
    RecoveryError,
    RecoveryStats,
    Transaction,
    TransactionManager,
    TransactionStats,
    WALManager,
)
from db_engine.ports.outbound import DiskManager, SyncMode, WALWriter

__all__ = [
    # Inbound ports
    "BufferPool",
    "BufferPoolFullError",
    "BufferPoolStats",
    "CheckpointData",
    "Index",
    "IndexKey",
    "IndexManager",
    "IndexMetadata",
    "IndexStats",
    "KeyType",
    "LockConflictError",
    "RecoveryError",
    "RecoveryStats",
    "Transaction",
    "TransactionManager",
    "TransactionStats",
    "WALManager",
    # Outbound ports
    "DiskManager",
    "SyncMode",
    "WALWriter",
]
