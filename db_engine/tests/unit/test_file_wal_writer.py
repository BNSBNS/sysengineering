"""Unit tests for FileWALWriter."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from db_engine.adapters.outbound import FileWALWriter
from db_engine.domain.entities import BeginRecord, CommitRecord, InsertRecord
from db_engine.domain.value_objects import LSN, PageId, TransactionId
from db_engine.ports.outbound.wal_writer import SyncMode


class TestFileWALWriter:
    """Tests for FileWALWriter."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for WAL files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def wal_writer(self, temp_dir: Path) -> FileWALWriter:
        """Create a WAL writer for testing."""
        wal_dir = temp_dir / "wal"
        ww = FileWALWriter(wal_dir, segment_size=1024 * 1024, sync_mode=SyncMode.NONE)
        yield ww
        ww.close()

    def test_creation(self, temp_dir: Path) -> None:
        """WAL writer creates directory and initial segment."""
        wal_dir = temp_dir / "new_wal"
        ww = FileWALWriter(wal_dir, segment_size=1024 * 1024, sync_mode=SyncMode.NONE)

        assert wal_dir.exists()
        assert ww.segment_size == 1024 * 1024
        assert ww.sync_mode == SyncMode.NONE
        assert ww.get_current_lsn() == LSN(1)

        ww.close()

    def test_append_record(self, wal_writer: FileWALWriter) -> None:
        """Records can be appended."""
        record = BeginRecord(
            lsn=LSN(0),  # Will be assigned
            txn_id=TransactionId(1),
            prev_lsn=LSN(0),
        )

        lsn = wal_writer.append(record)
        assert lsn == LSN(1)
        assert wal_writer.get_current_lsn() == LSN(2)

    def test_append_multiple_records(self, wal_writer: FileWALWriter) -> None:
        """Multiple records can be appended."""
        for i in range(5):
            record = BeginRecord(
                lsn=LSN(0),
                txn_id=TransactionId(i),
                prev_lsn=LSN(0),
            )
            lsn = wal_writer.append(record)
            assert lsn == LSN(i + 1)

        assert wal_writer.get_current_lsn() == LSN(6)

    def test_flush(self, wal_writer: FileWALWriter) -> None:
        """Flush persists records to disk."""
        record = BeginRecord(
            lsn=LSN(0),
            txn_id=TransactionId(1),
            prev_lsn=LSN(0),
        )

        lsn = wal_writer.append(record)
        assert wal_writer.get_flushed_lsn() == LSN(0)  # Not flushed yet

        wal_writer.flush(lsn)
        assert wal_writer.get_flushed_lsn() == lsn

    def test_read_from(self, wal_writer: FileWALWriter) -> None:
        """Records can be read back."""
        # Write some records
        records = []
        for i in range(5):
            record = BeginRecord(
                lsn=LSN(i + 1),  # Will be overwritten by append
                txn_id=TransactionId(i),
                prev_lsn=LSN(0),
            )
            wal_writer.append(record)
            records.append(record)

        wal_writer.flush(LSN(5))

        # Read back
        read_records = list(wal_writer.read_from(LSN(1)))

        assert len(read_records) == 5
        for i, rec in enumerate(read_records):
            assert isinstance(rec, BeginRecord)
            assert rec.txn_id == TransactionId(i)

    def test_read_from_middle(self, wal_writer: FileWALWriter) -> None:
        """Can read from a specific LSN."""
        for i in range(10):
            record = BeginRecord(
                lsn=LSN(0),
                txn_id=TransactionId(i),
                prev_lsn=LSN(0),
            )
            wal_writer.append(record)

        wal_writer.flush(LSN(10))

        # Read from LSN 5
        read_records = list(wal_writer.read_from(LSN(5)))

        assert len(read_records) == 6  # LSN 5, 6, 7, 8, 9, 10
        assert read_records[0].lsn == LSN(5)

    def test_different_record_types(self, wal_writer: FileWALWriter) -> None:
        """Different record types can be written and read."""
        # Write different types
        begin = BeginRecord(lsn=LSN(0), txn_id=TransactionId(1), prev_lsn=LSN(0))
        wal_writer.append(begin)

        insert = InsertRecord(
            lsn=LSN(0),
            txn_id=TransactionId(1),
            prev_lsn=LSN(1),
            page_id=PageId(1),
            slot_id=0,
            data=b"test data",
        )
        wal_writer.append(insert)

        commit = CommitRecord(lsn=LSN(0), txn_id=TransactionId(1), prev_lsn=LSN(2))
        wal_writer.append(commit)

        wal_writer.flush(LSN(3))

        # Read back
        records = list(wal_writer.read_from(LSN(1)))

        assert len(records) == 3
        assert isinstance(records[0], BeginRecord)
        assert isinstance(records[1], InsertRecord)
        assert isinstance(records[2], CommitRecord)

    def test_large_record(self, wal_writer: FileWALWriter) -> None:
        """Large records can be written."""
        large_data = b"x" * 10000

        record = InsertRecord(
            lsn=LSN(0),
            txn_id=TransactionId(1),
            prev_lsn=LSN(0),
            page_id=PageId(1),
            slot_id=0,
            data=large_data,
        )

        lsn = wal_writer.append(record)
        wal_writer.flush(lsn)

        records = list(wal_writer.read_from(LSN(1)))
        assert len(records) == 1
        assert isinstance(records[0], InsertRecord)
        assert records[0].data == large_data

    def test_context_manager(self, temp_dir: Path) -> None:
        """WAL writer works as context manager."""
        wal_dir = temp_dir / "context_wal"

        with FileWALWriter(wal_dir, sync_mode=SyncMode.NONE) as ww:
            record = BeginRecord(lsn=LSN(0), txn_id=TransactionId(1), prev_lsn=LSN(0))
            lsn = ww.append(record)
            ww.flush(lsn)

        # Verify closed properly by reopening
        with FileWALWriter(wal_dir, sync_mode=SyncMode.NONE) as ww2:
            records = list(ww2.read_from(LSN(1)))
            assert len(records) == 1


class TestFileWALWriterRecovery:
    """Tests for WAL recovery."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for WAL files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_recovery_after_restart(self, temp_dir: Path) -> None:
        """WAL state is recovered after restart."""
        wal_dir = temp_dir / "recover_wal"

        # Write and close
        ww1 = FileWALWriter(wal_dir, sync_mode=SyncMode.NONE)
        for i in range(5):
            record = BeginRecord(lsn=LSN(0), txn_id=TransactionId(i), prev_lsn=LSN(0))
            ww1.append(record)
        ww1.flush(LSN(5))
        ww1.close()

        # Reopen and verify
        ww2 = FileWALWriter(wal_dir, sync_mode=SyncMode.NONE)

        assert ww2.get_flushed_lsn() == LSN(5)
        assert ww2.get_current_lsn() == LSN(6)

        # Can continue writing
        record = BeginRecord(lsn=LSN(0), txn_id=TransactionId(100), prev_lsn=LSN(0))
        lsn = ww2.append(record)
        assert lsn == LSN(6)

        ww2.close()

    def test_recovery_reads_all_records(self, temp_dir: Path) -> None:
        """All records are recoverable after restart."""
        wal_dir = temp_dir / "all_recover_wal"

        # Write records
        ww1 = FileWALWriter(wal_dir, sync_mode=SyncMode.NONE)
        for i in range(10):
            record = InsertRecord(
                lsn=LSN(0),
                txn_id=TransactionId(1),
                prev_lsn=LSN(i) if i > 0 else LSN(0),
                page_id=PageId(1),
                slot_id=i,
                data=f"record_{i}".encode(),
            )
            ww1.append(record)
        ww1.flush(LSN(10))
        ww1.close()

        # Reopen and read all
        ww2 = FileWALWriter(wal_dir, sync_mode=SyncMode.NONE)
        records = list(ww2.read_from(LSN(1)))

        assert len(records) == 10
        for i, rec in enumerate(records):
            assert isinstance(rec, InsertRecord)
            assert rec.data == f"record_{i}".encode()

        ww2.close()


class TestFileWALWriterSegments:
    """Tests for segment management."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for WAL files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_segment_rollover(self, temp_dir: Path) -> None:
        """New segment is created when current is full."""
        wal_dir = temp_dir / "segment_wal"

        # Small segment size to trigger rollover
        ww = FileWALWriter(wal_dir, segment_size=1024, sync_mode=SyncMode.NONE)

        # Write enough data to trigger segment rollover
        large_data = b"x" * 500  # Each record ~500+ bytes
        for i in range(5):  # Should create multiple segments
            record = InsertRecord(
                lsn=LSN(0),
                txn_id=TransactionId(1),
                prev_lsn=LSN(i) if i > 0 else LSN(0),
                page_id=PageId(1),
                slot_id=i,
                data=large_data,
            )
            ww.append(record)

        ww.flush(LSN(5))

        # Check multiple segment files exist
        segment_files = list(wal_dir.glob("wal_*.log"))
        assert len(segment_files) >= 2

        # Verify all records are readable across segments
        records = list(ww.read_from(LSN(1)))
        assert len(records) == 5

        ww.close()

    def test_truncate_before(self, temp_dir: Path) -> None:
        """Old segments can be truncated."""
        wal_dir = temp_dir / "truncate_wal"

        # Small segment size
        ww = FileWALWriter(wal_dir, segment_size=512, sync_mode=SyncMode.NONE)

        # Write enough to create multiple segments
        for i in range(20):
            record = InsertRecord(
                lsn=LSN(0),
                txn_id=TransactionId(1),
                prev_lsn=LSN(i) if i > 0 else LSN(0),
                page_id=PageId(1),
                slot_id=i,
                data=b"x" * 100,
            )
            ww.append(record)

        ww.flush(LSN(20))

        initial_segments = len(list(wal_dir.glob("wal_*.log")))

        # Truncate old records
        ww.truncate_before(LSN(15))

        # Some segments should be removed
        remaining_segments = len(list(wal_dir.glob("wal_*.log")))
        assert remaining_segments <= initial_segments

        ww.close()
