"""Unit tests for B+Tree node structures."""

from __future__ import annotations

import pytest

from db_engine.domain.entities import (
    BTreeInternalNode,
    BTreeLeafNode,
    BTreeNodeHeader,
    NodeType,
)
from db_engine.domain.value_objects import INVALID_PAGE_ID, PageId, RecordId
from db_engine.ports.inbound.index_manager import IndexKey, KeyType


class TestBTreeNodeHeader:
    """Tests for BTreeNodeHeader."""

    def test_header_creation(self) -> None:
        """Header can be created with defaults."""
        header = BTreeNodeHeader(node_type=NodeType.LEAF)

        assert header.node_type == NodeType.LEAF
        assert header.num_keys == 0
        assert header.parent_page_id == INVALID_PAGE_ID
        assert header.next_page_id == INVALID_PAGE_ID
        assert header.prev_page_id == INVALID_PAGE_ID

    def test_header_serialization(self) -> None:
        """Header can be serialized and deserialized."""
        header = BTreeNodeHeader(
            node_type=NodeType.INTERNAL,
            num_keys=5,
            parent_page_id=PageId(1),
            next_page_id=PageId(2),
            prev_page_id=PageId(3),
        )

        data = header.to_bytes()
        assert len(data) == 24  # NODE_HEADER_SIZE

        restored = BTreeNodeHeader.from_bytes(data)
        assert restored.node_type == NodeType.INTERNAL
        assert restored.num_keys == 5
        assert restored.parent_page_id == PageId(1)
        assert restored.next_page_id == PageId(2)
        assert restored.prev_page_id == PageId(3)


class TestBTreeLeafNode:
    """Tests for BTreeLeafNode."""

    def test_leaf_creation(self) -> None:
        """Leaf node can be created."""
        leaf = BTreeLeafNode.new(PageId(0))

        assert leaf.page_id == PageId(0)
        assert leaf.is_leaf is True
        assert leaf.num_keys == 0
        assert len(leaf.keys) == 0
        assert len(leaf.values) == 0

    def test_leaf_insert(self) -> None:
        """Leaf node supports key insertion."""
        leaf = BTreeLeafNode.new(PageId(0))
        key1 = IndexKey(value=10, key_type=KeyType.INTEGER)
        key2 = IndexKey(value=5, key_type=KeyType.INTEGER)
        key3 = IndexKey(value=15, key_type=KeyType.INTEGER)
        rid1 = RecordId(PageId(1), 0)
        rid2 = RecordId(PageId(1), 1)
        rid3 = RecordId(PageId(1), 2)

        assert leaf.insert(key1, rid1) is True
        assert leaf.insert(key2, rid2) is True
        assert leaf.insert(key3, rid3) is True

        # Keys should be in sorted order
        assert leaf.num_keys == 3
        assert leaf.keys[0].value == 5
        assert leaf.keys[1].value == 10
        assert leaf.keys[2].value == 15

    def test_leaf_insert_duplicate(self) -> None:
        """Leaf node rejects duplicate keys."""
        leaf = BTreeLeafNode.new(PageId(0))
        key = IndexKey(value=10, key_type=KeyType.INTEGER)
        rid = RecordId(PageId(1), 0)

        assert leaf.insert(key, rid) is True
        assert leaf.insert(key, rid) is False  # Duplicate
        assert leaf.num_keys == 1

    def test_leaf_search(self) -> None:
        """Leaf node supports key search."""
        leaf = BTreeLeafNode.new(PageId(0))
        key1 = IndexKey(value=10, key_type=KeyType.INTEGER)
        key2 = IndexKey(value=20, key_type=KeyType.INTEGER)
        rid1 = RecordId(PageId(1), 0)
        rid2 = RecordId(PageId(1), 1)

        leaf.insert(key1, rid1)
        leaf.insert(key2, rid2)

        assert leaf.search(key1) == rid1
        assert leaf.search(key2) == rid2
        assert leaf.search(IndexKey(value=15, key_type=KeyType.INTEGER)) is None

    def test_leaf_delete(self) -> None:
        """Leaf node supports key deletion."""
        leaf = BTreeLeafNode.new(PageId(0))
        key1 = IndexKey(value=10, key_type=KeyType.INTEGER)
        key2 = IndexKey(value=20, key_type=KeyType.INTEGER)
        rid1 = RecordId(PageId(1), 0)
        rid2 = RecordId(PageId(1), 1)

        leaf.insert(key1, rid1)
        leaf.insert(key2, rid2)

        assert leaf.delete(key1) is True
        assert leaf.num_keys == 1
        assert leaf.search(key1) is None
        assert leaf.search(key2) == rid2

        assert leaf.delete(key1) is False  # Already deleted

    def test_leaf_get_min_key(self) -> None:
        """Leaf node returns minimum key."""
        leaf = BTreeLeafNode.new(PageId(0))
        assert leaf.get_min_key() is None

        key1 = IndexKey(value=10, key_type=KeyType.INTEGER)
        key2 = IndexKey(value=5, key_type=KeyType.INTEGER)
        rid1 = RecordId(PageId(1), 0)
        rid2 = RecordId(PageId(1), 1)

        leaf.insert(key1, rid1)
        assert leaf.get_min_key() == key1

        leaf.insert(key2, rid2)
        assert leaf.get_min_key() == key2  # 5 is now minimum


class TestBTreeInternalNode:
    """Tests for BTreeInternalNode."""

    def test_internal_creation(self) -> None:
        """Internal node can be created."""
        node = BTreeInternalNode.new(PageId(0))

        assert node.page_id == PageId(0)
        assert node.is_leaf is False
        assert node.num_keys == 0

    def test_internal_insert_child(self) -> None:
        """Internal node supports child insertion."""
        node = BTreeInternalNode.new(PageId(0))
        key = IndexKey(value=10, key_type=KeyType.INTEGER)

        node.insert_child(key, PageId(1), PageId(2))

        assert node.num_keys == 1
        assert len(node.children) == 2
        assert node.children[0] == PageId(1)
        assert node.children[1] == PageId(2)

    def test_internal_find_child(self) -> None:
        """Internal node finds correct child for key."""
        node = BTreeInternalNode.new(PageId(0))
        key1 = IndexKey(value=10, key_type=KeyType.INTEGER)
        key2 = IndexKey(value=20, key_type=KeyType.INTEGER)

        node.insert_child(key1, PageId(1), PageId(2))
        node.insert_child(key2, PageId(2), PageId(3))

        # Key < 10 -> child 0
        assert node.find_child(IndexKey(value=5, key_type=KeyType.INTEGER)) == PageId(1)
        # 10 <= Key < 20 -> child 1
        assert node.find_child(IndexKey(value=10, key_type=KeyType.INTEGER)) == PageId(2)
        assert node.find_child(IndexKey(value=15, key_type=KeyType.INTEGER)) == PageId(2)
        # Key >= 20 -> child 2
        assert node.find_child(IndexKey(value=20, key_type=KeyType.INTEGER)) == PageId(3)
        assert node.find_child(IndexKey(value=25, key_type=KeyType.INTEGER)) == PageId(3)

    def test_internal_get_min_key(self) -> None:
        """Internal node returns minimum key."""
        node = BTreeInternalNode.new(PageId(0))
        assert node.get_min_key() is None

        key1 = IndexKey(value=20, key_type=KeyType.INTEGER)
        node.insert_child(key1, PageId(1), PageId(2))
        assert node.get_min_key() == key1

        key2 = IndexKey(value=10, key_type=KeyType.INTEGER)
        node.insert_child(key2, PageId(0), PageId(1))
        assert node.get_min_key() == key2  # 10 is now minimum
