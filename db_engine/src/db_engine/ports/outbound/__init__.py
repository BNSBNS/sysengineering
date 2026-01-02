"""Outbound ports - interfaces for external dependencies.

Outbound ports define contracts for external systems that the
database engine depends on, such as disk I/O and WAL persistence.
"""

from db_engine.ports.outbound.disk_manager import DiskManager
from db_engine.ports.outbound.wal_writer import SyncMode, WALWriter

__all__ = [
    "DiskManager",
    "WALWriter",
    "SyncMode",
]
