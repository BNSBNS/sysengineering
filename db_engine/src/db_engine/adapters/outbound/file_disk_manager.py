"""File-based Disk Manager implementation.

This adapter implements the DiskManager protocol using standard file I/O.
It manages a single data file containing fixed-size pages.

File Format:
    - Header page (page 0): metadata (page count, page size, etc.)
    - Data pages (page 1+): actual database pages

Thread Safety:
    Read operations are thread-safe. Write operations to different pages
    can be concurrent, but writes to the same page must be externally
    synchronized (typically by the buffer pool).

References:
    - design.md Section 2 (Disk Layer)
"""

from __future__ import annotations

import os
import struct
import threading
from pathlib import Path
from typing import BinaryIO

from db_engine.domain.value_objects import PageId, INVALID_PAGE_ID
from db_engine.infrastructure.config import get_config


# Header page format (page 0)
# Magic (8 bytes) + Version (4 bytes) + Page Size (4 bytes) + Page Count (4 bytes) + Reserved (4076 bytes)
HEADER_MAGIC = b"DBENGINE"
HEADER_VERSION = 1
HEADER_FORMAT = ">8sIII"  # magic, version, page_size, page_count
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class FileDiskManager:
    """File-based implementation of the DiskManager protocol.

    Manages a single database file with fixed-size pages. The first page
    (page 0) is reserved for metadata.

    Attributes:
        file_path: Path to the database file.
        page_size: Size of each page in bytes.
    """

    def __init__(
        self,
        file_path: str | Path,
        page_size: int | None = None,
        create: bool = True,
    ) -> None:
        """Initialize the disk manager.

        Args:
            file_path: Path to the database file.
            page_size: Size of each page (default from config).
            create: If True, create the file if it doesn't exist.

        Raises:
            FileNotFoundError: If file doesn't exist and create=False.
            ValueError: If existing file has incompatible page size.
        """
        self._file_path = Path(file_path)
        self._page_size = page_size or get_config().storage.page_size
        self._file: BinaryIO | None = None
        self._page_count = 0
        self._lock = threading.Lock()
        self._closed = False

        # Free page list (simple approach - track deallocated pages)
        self._free_pages: list[PageId] = []

        if self._file_path.exists():
            self._open_existing()
        elif create:
            self._create_new()
        else:
            raise FileNotFoundError(f"Database file not found: {self._file_path}")

    def _create_new(self) -> None:
        """Create a new database file with header page."""
        # Ensure parent directory exists
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

        self._file = open(self._file_path, "w+b")

        # Write header page
        header = struct.pack(
            HEADER_FORMAT,
            HEADER_MAGIC,
            HEADER_VERSION,
            self._page_size,
            1,  # page_count = 1 (just the header page)
        )
        # Pad to full page size
        header_page = header + b"\x00" * (self._page_size - len(header))
        self._file.write(header_page)
        self._file.flush()
        os.fsync(self._file.fileno())

        self._page_count = 1

    def _open_existing(self) -> None:
        """Open an existing database file and validate header."""
        self._file = open(self._file_path, "r+b")

        # Read and validate header
        header_data = self._file.read(HEADER_SIZE)
        if len(header_data) < HEADER_SIZE:
            raise ValueError("Invalid database file: header too short")

        magic, version, page_size, page_count = struct.unpack(
            HEADER_FORMAT, header_data
        )

        if magic != HEADER_MAGIC:
            raise ValueError(f"Invalid database file: bad magic {magic!r}")

        if version != HEADER_VERSION:
            raise ValueError(f"Unsupported database version: {version}")

        if page_size != self._page_size:
            raise ValueError(
                f"Page size mismatch: file has {page_size}, expected {self._page_size}"
            )

        self._page_count = page_count

    def _update_header(self) -> None:
        """Update the header page with current metadata."""
        if self._file is None:
            return

        header = struct.pack(
            HEADER_FORMAT,
            HEADER_MAGIC,
            HEADER_VERSION,
            self._page_size,
            self._page_count,
        )

        self._file.seek(0)
        self._file.write(header)

    @property
    def page_size(self) -> int:
        """Return the fixed page size in bytes."""
        return self._page_size

    def read_page(self, page_id: PageId) -> bytes:
        """Read a page from disk.

        Args:
            page_id: The page to read.

        Returns:
            Raw page data as bytes (exactly page_size bytes).

        Raises:
            ValueError: If page_id is invalid.
            IOError: If the read fails.
        """
        if self._closed or self._file is None:
            raise IOError("Disk manager is closed")

        if page_id < 0 or page_id >= self._page_count:
            raise ValueError(f"Invalid page_id: {page_id} (max: {self._page_count - 1})")

        # Page 0 is the header, user pages start at 1
        offset = page_id * self._page_size

        with self._lock:
            self._file.seek(offset)
            data = self._file.read(self._page_size)

        if len(data) != self._page_size:
            raise IOError(f"Short read: got {len(data)} bytes, expected {self._page_size}")

        return data

    def write_page(self, page_id: PageId, data: bytes) -> None:
        """Write a page to disk.

        Args:
            page_id: The page to write.
            data: Raw page data (must be exactly page_size bytes).

        Raises:
            ValueError: If page_id is invalid or data size is wrong.
            IOError: If the write fails.
        """
        if self._closed or self._file is None:
            raise IOError("Disk manager is closed")

        if page_id < 1:  # Can't write to header page directly
            raise ValueError(f"Cannot write to page {page_id} (reserved)")

        if page_id >= self._page_count:
            raise ValueError(f"Invalid page_id: {page_id} (max: {self._page_count - 1})")

        if len(data) != self._page_size:
            raise ValueError(
                f"Data size mismatch: got {len(data)}, expected {self._page_size}"
            )

        offset = page_id * self._page_size

        with self._lock:
            self._file.seek(offset)
            self._file.write(data)

    def allocate_page(self) -> PageId:
        """Allocate a new page and return its ID.

        Returns:
            The PageId of the newly allocated page.

        Raises:
            IOError: If allocation fails.
        """
        if self._closed or self._file is None:
            raise IOError("Disk manager is closed")

        with self._lock:
            # Check if we have a free page to reuse
            if self._free_pages:
                page_id = self._free_pages.pop()
                return page_id

            # Allocate a new page at the end
            page_id = PageId(self._page_count)
            self._page_count += 1

            # Extend the file with an empty page
            self._file.seek(page_id * self._page_size)
            self._file.write(b"\x00" * self._page_size)

            # Update header
            self._update_header()

            return page_id

    def deallocate_page(self, page_id: PageId) -> None:
        """Mark a page as free for reuse.

        Args:
            page_id: The page to deallocate.

        Raises:
            ValueError: If page_id is invalid or already free.
        """
        if self._closed:
            raise IOError("Disk manager is closed")

        if page_id < 1:
            raise ValueError(f"Cannot deallocate page {page_id} (reserved)")

        if page_id >= self._page_count:
            raise ValueError(f"Invalid page_id: {page_id}")

        with self._lock:
            if page_id in self._free_pages:
                raise ValueError(f"Page {page_id} is already free")

            self._free_pages.append(page_id)

    def get_num_pages(self) -> int:
        """Return the total number of allocated pages.

        Returns:
            The count of pages currently allocated (excluding header).
        """
        # Subtract 1 for header page
        return max(0, self._page_count - 1)

    def sync(self) -> None:
        """Ensure all writes are persisted to stable storage."""
        if self._closed or self._file is None:
            return

        with self._lock:
            self._file.flush()
            os.fsync(self._file.fileno())

    def close(self) -> None:
        """Close the disk manager and release resources."""
        if self._closed:
            return

        with self._lock:
            self._closed = True
            if self._file is not None:
                # Update header with final page count
                self._update_header()
                self._file.flush()
                os.fsync(self._file.fileno())
                self._file.close()
                self._file = None

    def __enter__(self) -> FileDiskManager:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __del__(self) -> None:
        """Destructor - ensure file is closed."""
        self.close()
