"""Inbound ports - API contracts for the database engine.

Inbound ports define the interfaces that clients and upper layers
use to interact with the storage engine, transaction manager, etc.
"""

from db_engine.ports.inbound.buffer_pool import (
    BufferPool,
    BufferPoolFullError,
    BufferPoolStats,
)
from db_engine.ports.inbound.index_manager import (
    Index,
    IndexKey,
    IndexManager,
    IndexMetadata,
    IndexStats,
    KeyType,
)
from db_engine.ports.inbound.transaction_manager import (
    LockConflictError,
    Transaction,
    TransactionManager,
    TransactionStats,
)
from db_engine.ports.inbound.wal_manager import (
    CheckpointData,
    RecoveryError,
    RecoveryStats,
    WALManager,
)

__all__ = [
    # Buffer Pool
    "BufferPool",
    "BufferPoolFullError",
    "BufferPoolStats",
    # Index Manager
    "Index",
    "IndexKey",
    "IndexManager",
    "IndexMetadata",
    "IndexStats",
    "KeyType",
    # Transaction Manager
    "LockConflictError",
    "Transaction",
    "TransactionManager",
    "TransactionStats",
    # WAL Manager
    "CheckpointData",
    "RecoveryError",
    "RecoveryStats",
    "WALManager",
]
