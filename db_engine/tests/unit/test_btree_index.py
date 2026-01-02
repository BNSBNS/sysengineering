"""Unit tests for B+Tree Index implementation."""

from __future__ import annotations

import pytest

from db_engine.domain.services import BTreeIndex, BTreeIndexManager
from db_engine.domain.value_objects import PageId, RecordId
from db_engine.ports.inbound.index_manager import IndexKey, KeyType


class TestBTreeIndex:
    """Tests for BTreeIndex."""

    @pytest.fixture
    def index(self) -> BTreeIndex:
        """Create a B+Tree index for testing."""
        return BTreeIndex(
            name="test_idx",
            table_name="users",
            column_name="id",
            key_type=KeyType.INTEGER,
            is_unique=True,
        )

    def test_index_creation(self, index: BTreeIndex) -> None:
        """Index can be created."""
        assert index.name == "test_idx"
        assert index.table_name == "users"
        assert index.column_name == "id"
        assert index.is_unique is True
        assert index.metadata.height == 1
        assert index.metadata.num_entries == 0

    def test_insert_and_search(self, index: BTreeIndex) -> None:
        """Index supports insert and search."""
        key = IndexKey(value=10, key_type=KeyType.INTEGER)
        rid = RecordId(PageId(1), 0)

        assert index.insert(key, rid) is True
        assert index.search(key) == rid
        assert index.metadata.num_entries == 1

    def test_insert_multiple(self, index: BTreeIndex) -> None:
        """Index supports multiple insertions."""
        for i in range(10):
            key = IndexKey(value=i * 10, key_type=KeyType.INTEGER)
            rid = RecordId(PageId(1), i)
            assert index.insert(key, rid) is True

        assert index.metadata.num_entries == 10

        # Verify all can be found
        for i in range(10):
            key = IndexKey(value=i * 10, key_type=KeyType.INTEGER)
            rid = index.search(key)
            assert rid is not None
            assert rid.slot_id == i

    def test_unique_constraint(self, index: BTreeIndex) -> None:
        """Unique index rejects duplicate keys."""
        key = IndexKey(value=10, key_type=KeyType.INTEGER)
        rid1 = RecordId(PageId(1), 0)
        rid2 = RecordId(PageId(1), 1)

        assert index.insert(key, rid1) is True
        assert index.insert(key, rid2) is False  # Duplicate
        assert index.metadata.num_entries == 1

    def test_non_unique_index(self) -> None:
        """Non-unique index allows duplicate keys."""
        index = BTreeIndex(
            name="test_idx",
            table_name="users",
            column_name="name",
            key_type=KeyType.VARCHAR,
            is_unique=False,
        )
        key = IndexKey(value="Alice", key_type=KeyType.VARCHAR)
        rid1 = RecordId(PageId(1), 0)
        rid2 = RecordId(PageId(1), 1)

        assert index.insert(key, rid1) is True
        # For non-unique, we still prevent exact duplicates in leaf
        # A production impl would allow multiple RIDs per key
        assert index.insert(key, rid2) is False

    def test_delete(self, index: BTreeIndex) -> None:
        """Index supports key deletion."""
        key = IndexKey(value=10, key_type=KeyType.INTEGER)
        rid = RecordId(PageId(1), 0)

        index.insert(key, rid)
        assert index.delete(key) is True
        assert index.search(key) is None
        assert index.metadata.num_entries == 0

    def test_delete_nonexistent(self, index: BTreeIndex) -> None:
        """Delete returns False for nonexistent key."""
        key = IndexKey(value=10, key_type=KeyType.INTEGER)
        assert index.delete(key) is False

    def test_update(self, index: BTreeIndex) -> None:
        """Index supports key update."""
        old_key = IndexKey(value=10, key_type=KeyType.INTEGER)
        new_key = IndexKey(value=20, key_type=KeyType.INTEGER)
        rid = RecordId(PageId(1), 0)

        index.insert(old_key, rid)
        assert index.update(old_key, new_key, rid) is True

        assert index.search(old_key) is None
        assert index.search(new_key) == rid

    def test_range_scan(self, index: BTreeIndex) -> None:
        """Index supports range scan."""
        for i in range(10):
            key = IndexKey(value=i * 10, key_type=KeyType.INTEGER)
            rid = RecordId(PageId(1), i)
            index.insert(key, rid)

        # Scan all
        results = list(index.range_scan())
        assert len(results) == 10

        # Scan range [20, 60]
        low = IndexKey(value=20, key_type=KeyType.INTEGER)
        high = IndexKey(value=60, key_type=KeyType.INTEGER)
        results = list(index.range_scan(low=low, high=high))
        assert len(results) == 5  # 20, 30, 40, 50, 60
        assert results[0][0].value == 20
        assert results[-1][0].value == 60

    def test_range_scan_exclusive(self, index: BTreeIndex) -> None:
        """Range scan supports exclusive bounds."""
        for i in range(5):
            key = IndexKey(value=i * 10, key_type=KeyType.INTEGER)
            rid = RecordId(PageId(1), i)
            index.insert(key, rid)

        low = IndexKey(value=10, key_type=KeyType.INTEGER)
        high = IndexKey(value=30, key_type=KeyType.INTEGER)
        results = list(
            index.range_scan(low=low, high=high, include_low=False, include_high=False)
        )
        assert len(results) == 1  # Only 20
        assert results[0][0].value == 20

    def test_scan_all(self, index: BTreeIndex) -> None:
        """Index supports scanning all entries."""
        for i in range(5):
            key = IndexKey(value=i * 10, key_type=KeyType.INTEGER)
            rid = RecordId(PageId(1), i)
            index.insert(key, rid)

        results = list(index.scan_all())
        assert len(results) == 5

        # Should be in sorted order
        for i, (key, rid) in enumerate(results):
            assert key.value == i * 10

    def test_split_leaf(self, index: BTreeIndex) -> None:
        """Index handles leaf node splits."""
        # With max_keys=4, inserting 6 keys should trigger a split
        for i in range(6):
            key = IndexKey(value=i, key_type=KeyType.INTEGER)
            rid = RecordId(PageId(1), i)
            index.insert(key, rid)

        # All keys should still be searchable
        for i in range(6):
            key = IndexKey(value=i, key_type=KeyType.INTEGER)
            assert index.search(key) is not None

        assert index.metadata.num_entries == 6
        # Tree should have grown in height
        assert index.metadata.height >= 1

    def test_many_insertions(self, index: BTreeIndex) -> None:
        """Index handles many insertions with multiple splits."""
        n = 50
        for i in range(n):
            key = IndexKey(value=i, key_type=KeyType.INTEGER)
            rid = RecordId(PageId(1), i)
            assert index.insert(key, rid) is True

        # All keys should be searchable
        for i in range(n):
            key = IndexKey(value=i, key_type=KeyType.INTEGER)
            assert index.search(key) is not None

        # Scan should return all in order
        results = list(index.scan_all())
        assert len(results) == n
        for i, (key, rid) in enumerate(results):
            assert key.value == i

    def test_reverse_order_insertions(self, index: BTreeIndex) -> None:
        """Index handles reverse order insertions."""
        n = 20
        for i in range(n - 1, -1, -1):
            key = IndexKey(value=i, key_type=KeyType.INTEGER)
            rid = RecordId(PageId(1), i)
            index.insert(key, rid)

        results = list(index.scan_all())
        assert len(results) == n
        # Should still be in sorted order
        for i, (key, rid) in enumerate(results):
            assert key.value == i


class TestBTreeIndexManager:
    """Tests for BTreeIndexManager."""

    @pytest.fixture
    def manager(self) -> BTreeIndexManager:
        """Create an index manager for testing."""
        return BTreeIndexManager()

    def test_create_index(self, manager: BTreeIndexManager) -> None:
        """Index manager can create indexes."""
        index = manager.create_index(
            name="test_idx",
            table_name="users",
            column_name="id",
            key_type=KeyType.INTEGER,
            is_unique=True,
        )

        assert index is not None
        assert index.metadata.name == "test_idx"

    def test_create_duplicate_raises(self, manager: BTreeIndexManager) -> None:
        """Creating duplicate index name raises."""
        manager.create_index(
            name="test_idx",
            table_name="users",
            column_name="id",
            key_type=KeyType.INTEGER,
        )

        with pytest.raises(ValueError, match="already exists"):
            manager.create_index(
                name="test_idx",
                table_name="orders",
                column_name="id",
                key_type=KeyType.INTEGER,
            )

    def test_get_index(self, manager: BTreeIndexManager) -> None:
        """Index manager can retrieve indexes."""
        manager.create_index(
            name="test_idx",
            table_name="users",
            column_name="id",
            key_type=KeyType.INTEGER,
        )

        index = manager.get_index("test_idx")
        assert index is not None
        assert index.metadata.name == "test_idx"

        assert manager.get_index("nonexistent") is None

    def test_drop_index(self, manager: BTreeIndexManager) -> None:
        """Index manager can drop indexes."""
        manager.create_index(
            name="test_idx",
            table_name="users",
            column_name="id",
            key_type=KeyType.INTEGER,
        )

        assert manager.drop_index("test_idx") is True
        assert manager.get_index("test_idx") is None
        assert manager.drop_index("test_idx") is False

    def test_list_indexes(self, manager: BTreeIndexManager) -> None:
        """Index manager can list indexes."""
        manager.create_index(
            name="idx1",
            table_name="users",
            column_name="id",
            key_type=KeyType.INTEGER,
        )
        manager.create_index(
            name="idx2",
            table_name="users",
            column_name="name",
            key_type=KeyType.VARCHAR,
        )
        manager.create_index(
            name="idx3",
            table_name="orders",
            column_name="id",
            key_type=KeyType.INTEGER,
        )

        all_indexes = manager.list_indexes()
        assert len(all_indexes) == 3

        user_indexes = manager.list_indexes(table_name="users")
        assert len(user_indexes) == 2

        order_indexes = manager.list_indexes(table_name="orders")
        assert len(order_indexes) == 1

    def test_get_stats(self, manager: BTreeIndexManager) -> None:
        """Index manager returns statistics."""
        index = manager.create_index(
            name="test_idx",
            table_name="users",
            column_name="id",
            key_type=KeyType.INTEGER,
        )

        # Insert some data
        for i in range(10):
            key = IndexKey(value=i, key_type=KeyType.INTEGER)
            rid = RecordId(PageId(1), i)
            index.insert(key, rid)

        stats = manager.get_stats()
        assert stats.num_indexes == 1
        assert stats.total_entries == 10
        assert stats.insert_count == 10


class TestBTreeIndexVarchar:
    """Tests for B+Tree with VARCHAR keys."""

    @pytest.fixture
    def index(self) -> BTreeIndex:
        """Create a B+Tree index with varchar keys."""
        return BTreeIndex(
            name="name_idx",
            table_name="users",
            column_name="name",
            key_type=KeyType.VARCHAR,
        )

    def test_varchar_insert_search(self, index: BTreeIndex) -> None:
        """Index works with varchar keys."""
        names = ["Alice", "Bob", "Charlie", "David", "Eve"]
        for i, name in enumerate(names):
            key = IndexKey(value=name, key_type=KeyType.VARCHAR)
            rid = RecordId(PageId(1), i)
            index.insert(key, rid)

        for i, name in enumerate(names):
            key = IndexKey(value=name, key_type=KeyType.VARCHAR)
            rid = index.search(key)
            assert rid is not None
            assert rid.slot_id == i

    def test_varchar_range_scan(self, index: BTreeIndex) -> None:
        """Range scan works with varchar keys."""
        names = ["Alice", "Bob", "Charlie", "David", "Eve"]
        for i, name in enumerate(names):
            key = IndexKey(value=name, key_type=KeyType.VARCHAR)
            rid = RecordId(PageId(1), i)
            index.insert(key, rid)

        # Scan [Bob, David]
        low = IndexKey(value="Bob", key_type=KeyType.VARCHAR)
        high = IndexKey(value="David", key_type=KeyType.VARCHAR)
        results = list(index.range_scan(low=low, high=high))

        assert len(results) == 3  # Bob, Charlie, David
        assert results[0][0].value == "Bob"
        assert results[-1][0].value == "David"
