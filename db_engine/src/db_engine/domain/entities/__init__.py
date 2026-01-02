"""Domain entities for the database engine.

Entities are objects with identity that have a lifecycle. Unlike value objects,
two entities with the same attributes may not be equal if they have different
identities.

Exports:
    Page:
        - PageHeader: 24-byte header with page metadata
        - Slot: Slot array entry (offset, length)
        - SlottedPage: Variable-length record storage

    Record:
        - RecordHeader: MVCC metadata (xmin, xmax, LSNs)
        - Record: Data record with MVCC header
        - Snapshot: Point-in-time view for MVCC visibility

    WAL Records:
        - LogRecord: Base class for all WAL records
        - LogRecordType: Enumeration of record types
        - BeginRecord, CommitRecord, AbortRecord: Transaction lifecycle
        - UpdateRecord, InsertRecord, DeleteRecord: Data modification
        - CLRRecord: Compensation for undo
        - CheckpointRecord: Recovery optimization

    B+Tree Nodes:
        - BTreeNodeHeader: Node header with metadata
        - BTreeLeafNode: Leaf node storing key-RID pairs
        - BTreeInternalNode: Internal node with separator keys
        - NodeType: Enum for node types
"""

from db_engine.domain.entities.btree_node import (
    BTreeInternalNode,
    BTreeLeafNode,
    BTreeNodeHeader,
    NodeType,
)
from db_engine.domain.entities.page import PageHeader, Slot, SlottedPage
from db_engine.domain.entities.record import Record, RecordHeader, Snapshot
from db_engine.domain.entities.wal_record import (
    AbortRecord,
    BeginRecord,
    CheckpointRecord,
    CLRRecord,
    CommitRecord,
    DeleteRecord,
    InsertRecord,
    LogRecord,
    LogRecordType,
    UpdateRecord,
)

__all__ = [
    # Page
    "PageHeader",
    "Slot",
    "SlottedPage",
    # Record
    "RecordHeader",
    "Record",
    "Snapshot",
    # WAL Records
    "LogRecord",
    "LogRecordType",
    "BeginRecord",
    "CommitRecord",
    "AbortRecord",
    "UpdateRecord",
    "InsertRecord",
    "DeleteRecord",
    "CLRRecord",
    "CheckpointRecord",
    # B+Tree Nodes
    "BTreeNodeHeader",
    "BTreeLeafNode",
    "BTreeInternalNode",
    "NodeType",
]
