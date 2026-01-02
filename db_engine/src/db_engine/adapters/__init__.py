"""Adapters layer - concrete implementations of port interfaces.

Adapters provide the actual implementations:
- Inbound adapters: Handle incoming requests (gRPC, REST, CLI)
- Outbound adapters: Implement external dependencies (disk, network)
"""

from db_engine.adapters.outbound import (
    FileDiskManager,
    FileWALWriter,
    LRUBufferPool,
)

__all__ = [
    # Outbound adapters
    "FileDiskManager",
    "FileWALWriter",
    "LRUBufferPool",
]
