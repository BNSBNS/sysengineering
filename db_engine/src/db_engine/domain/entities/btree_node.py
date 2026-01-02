"""B+Tree node structures for index implementation.

This module defines the node structures for B+Tree indexes. A B+Tree is
a balanced tree optimized for disk-based storage with high fanout.

Key properties:
    - All data (RecordIds) stored in leaf nodes
    - Internal nodes only contain keys and child pointers
    - Leaf nodes are linked for efficient range scans
    - High fanout (100-1000) minimizes tree height

References:
    - Bayer & McCreight, "B+Trees" (1972)
    - design.md Section 3 (Index Manager)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

from db_engine.domain.value_objects import INVALID_PAGE_ID, PageId, RecordId

if TYPE_CHECKING:
    from db_engine.ports.inbound.index_manager import IndexKey


class NodeType(IntEnum):
    """Type of B+Tree node."""

    INTERNAL = 0
    LEAF = 1


# B+Tree node header format (24 bytes):
# - node_type: 1 byte
# - num_keys: 2 bytes (number of keys in node)
# - parent_page_id: 4 bytes
# - next_page_id: 4 bytes (for leaf nodes, sibling pointer)
# - prev_page_id: 4 bytes (for leaf nodes, sibling pointer)
# - reserved: 9 bytes (for alignment and future use)
NODE_HEADER_SIZE = 24
NODE_HEADER_FORMAT = "<BHiii9x"  # type, num_keys, parent, next, prev


@dataclass
class BTreeNodeHeader:
    """Header for a B+Tree node.

    Attributes:
        node_type: Whether this is an internal or leaf node.
        num_keys: Number of keys currently stored in this node.
        parent_page_id: Page ID of parent node (INVALID_PAGE_ID for root).
        next_page_id: For leaf nodes, the next sibling (INVALID_PAGE_ID if last).
        prev_page_id: For leaf nodes, the previous sibling (INVALID_PAGE_ID if first).
    """

    node_type: NodeType
    num_keys: int = 0
    parent_page_id: PageId = INVALID_PAGE_ID
    next_page_id: PageId = INVALID_PAGE_ID
    prev_page_id: PageId = INVALID_PAGE_ID

    def to_bytes(self) -> bytes:
        """Serialize header to bytes."""
        return struct.pack(
            NODE_HEADER_FORMAT,
            self.node_type,
            self.num_keys,
            self.parent_page_id,
            self.next_page_id,
            self.prev_page_id,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> BTreeNodeHeader:
        """Deserialize header from bytes."""
        node_type, num_keys, parent, next_id, prev_id = struct.unpack(
            NODE_HEADER_FORMAT, data[:NODE_HEADER_SIZE]
        )
        return cls(
            node_type=NodeType(node_type),
            num_keys=num_keys,
            parent_page_id=PageId(parent),
            next_page_id=PageId(next_id),
            prev_page_id=PageId(prev_id),
        )


@dataclass
class BTreeLeafNode:
    """A leaf node in a B+Tree.

    Leaf nodes store the actual key-value pairs where values are RecordIds
    pointing to the data pages. Leaf nodes are linked together for efficient
    range scans.

    Layout after header:
        [key1][rid1][key2][rid2]...[keyN][ridN]

    Where each key is a variable-length encoded value and each RID is 6 bytes.

    Attributes:
        page_id: The page ID of this node.
        header: Node header with metadata.
        keys: List of keys stored in this node.
        values: List of RecordIds corresponding to keys.
    """

    page_id: PageId
    header: BTreeNodeHeader
    keys: list[IndexKey] = field(default_factory=list)
    values: list[RecordId] = field(default_factory=list)

    @classmethod
    def new(cls, page_id: PageId) -> BTreeLeafNode:
        """Create a new empty leaf node."""
        return cls(
            page_id=page_id,
            header=BTreeNodeHeader(node_type=NodeType.LEAF),
            keys=[],
            values=[],
        )

    @property
    def is_leaf(self) -> bool:
        """Return True since this is a leaf node."""
        return True

    @property
    def num_keys(self) -> int:
        """Return the number of keys in this node."""
        return len(self.keys)

    def search(self, key: IndexKey) -> RecordId | None:
        """Search for a key in this leaf node.

        Args:
            key: The key to search for.

        Returns:
            The RecordId if found, None otherwise.
        """
        for i, k in enumerate(self.keys):
            if k == key:
                return self.values[i]
            if k > key:
                break
        return None

    def insert(self, key: IndexKey, rid: RecordId) -> bool:
        """Insert a key-value pair into this leaf node.

        Maintains sorted order of keys.

        Args:
            key: The key to insert.
            rid: The RecordId to associate with the key.

        Returns:
            True if inserted, False if key already exists.
        """
        # Find insertion position
        pos = 0
        for i, k in enumerate(self.keys):
            if k == key:
                return False  # Duplicate key
            if k > key:
                pos = i
                break
            pos = i + 1

        self.keys.insert(pos, key)
        self.values.insert(pos, rid)
        self.header.num_keys = len(self.keys)
        return True

    def delete(self, key: IndexKey) -> bool:
        """Delete a key from this leaf node.

        Args:
            key: The key to delete.

        Returns:
            True if deleted, False if key not found.
        """
        for i, k in enumerate(self.keys):
            if k == key:
                self.keys.pop(i)
                self.values.pop(i)
                self.header.num_keys = len(self.keys)
                return True
        return False

    def get_min_key(self) -> IndexKey | None:
        """Return the minimum key in this node."""
        if self.keys:
            return self.keys[0]
        return None


@dataclass
class BTreeInternalNode:
    """An internal node in a B+Tree.

    Internal nodes store keys and child page pointers. A node with N keys
    has N+1 children. Keys act as separators: all keys in child[i] are
    less than keys[i], and all keys in child[i+1] are >= keys[i].

    Layout after header:
        [child0][key1][child1][key2][child2]...[keyN][childN]

    Where each child is a 4-byte page ID and each key is variable-length.

    Attributes:
        page_id: The page ID of this node.
        header: Node header with metadata.
        keys: List of separator keys (N keys).
        children: List of child page IDs (N+1 children).
    """

    page_id: PageId
    header: BTreeNodeHeader
    keys: list[IndexKey] = field(default_factory=list)
    children: list[PageId] = field(default_factory=list)

    @classmethod
    def new(cls, page_id: PageId) -> BTreeInternalNode:
        """Create a new empty internal node."""
        return cls(
            page_id=page_id,
            header=BTreeNodeHeader(node_type=NodeType.INTERNAL),
            keys=[],
            children=[],
        )

    @property
    def is_leaf(self) -> bool:
        """Return False since this is an internal node."""
        return False

    @property
    def num_keys(self) -> int:
        """Return the number of keys in this node."""
        return len(self.keys)

    def find_child(self, key: IndexKey) -> PageId:
        """Find the child page ID for a given key.

        Args:
            key: The key to search for.

        Returns:
            The page ID of the child that should contain the key.
        """
        for i, k in enumerate(self.keys):
            if key < k:
                return self.children[i]
        return self.children[-1]

    def insert_child(self, key: IndexKey, left_child: PageId, right_child: PageId) -> None:
        """Insert a new separator key with its children.

        Called when a child splits. The left_child already exists; we're
        adding the new key and right_child.

        Args:
            key: The separator key (minimum key in right_child).
            left_child: The existing child page.
            right_child: The new child page (from split).
        """
        if not self.children:
            # First insertion
            self.children = [left_child, right_child]
            self.keys = [key]
            self.header.num_keys = 1
            return

        # Find position to insert
        pos = 0
        for i, k in enumerate(self.keys):
            if key < k:
                pos = i
                break
            pos = i + 1

        self.keys.insert(pos, key)
        # Insert right_child after the position
        self.children.insert(pos + 1, right_child)
        self.header.num_keys = len(self.keys)

    def get_min_key(self) -> IndexKey | None:
        """Return the minimum key in this node."""
        if self.keys:
            return self.keys[0]
        return None


# Union type for B+Tree nodes
BTreeNode = BTreeLeafNode | BTreeInternalNode
