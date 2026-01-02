"""Value objects for the database engine domain.

Value objects are immutable types that represent domain concepts.
They have no identity - two value objects with the same attributes are equal.

Exports:
    Identifiers:
        - PageId: Type-safe page identifier
        - TransactionId: Type-safe transaction identifier
        - LSN: Log Sequence Number for WAL records
        - RecordId: Composite identifier (page_id, slot_id) for records
        - INVALID_PAGE_ID, INVALID_TXN_ID, INVALID_LSN: Sentinel values

    Transaction Types:
        - TransactionState: Transaction lifecycle states (ACTIVE, COMMITTED, etc.)
        - IsolationLevel: Transaction isolation levels (SNAPSHOT, SERIALIZABLE, etc.)
        - LockMode: Lock modes for 2PL (SHARED, EXCLUSIVE, etc.)
        - WaitPolicy: Lock conflict handling policies
"""

from db_engine.domain.value_objects.identifiers import (
    INVALID_LSN,
    INVALID_PAGE_ID,
    INVALID_TXN_ID,
    LSN,
    PageId,
    RECORD_ID_SIZE,
    RecordId,
    TransactionId,
)
from db_engine.domain.value_objects.transaction_types import (
    IsolationLevel,
    LockMode,
    TransactionState,
    WaitPolicy,
)

__all__ = [
    # Identifiers
    "PageId",
    "TransactionId",
    "LSN",
    "RecordId",
    "INVALID_PAGE_ID",
    "INVALID_TXN_ID",
    "INVALID_LSN",
    "RECORD_ID_SIZE",
    # Transaction types
    "TransactionState",
    "IsolationLevel",
    "LockMode",
    "WaitPolicy",
]
