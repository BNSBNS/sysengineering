"""Outbound adapters - implementations of outbound ports.

These adapters implement external dependencies like disk I/O,
WAL persistence, and other infrastructure concerns.
"""

from db_engine.adapters.outbound.file_disk_manager import FileDiskManager
from db_engine.adapters.outbound.file_wal_writer import FileWALWriter
from db_engine.adapters.outbound.lru_buffer_pool import LRUBufferPool

__all__ = [
    "FileDiskManager",
    "FileWALWriter",
    "LRUBufferPool",
]
