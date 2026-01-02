"""B+Tree Index implementation.

This module implements a B+Tree index that provides efficient key-based
lookup and range scan operations. The B+Tree is optimized for disk-based
storage with high fanout to minimize tree height.

Key features:
    - O(log n) search, insert, delete
    - Efficient range scans via linked leaf nodes
    - Supports unique and non-unique indexes
    - Coordinates with buffer pool for page access

References:
    - Bayer & McCreight, "B+Trees" (1972)
    - design.md Section 3 (Index Manager)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

from db_engine.domain.entities.btree_node import (
    BTreeInternalNode,
    BTreeLeafNode,
    BTreeNode,
    NodeType,
)
from db_engine.domain.value_objects import INVALID_PAGE_ID, PageId, RecordId
from db_engine.ports.inbound.index_manager import (
    Index,
    IndexKey,
    IndexMetadata,
    IndexStats,
    KeyType,
)

if TYPE_CHECKING:
    from db_engine.ports.inbound.buffer_pool import BufferPool


# Default maximum keys per node (fanout - 1)
# In practice this would be calculated based on page size and key size
DEFAULT_MAX_KEYS = 4  # Small for testing; production would be much higher


@dataclass
class BTreeIndex:
    """A B+Tree index implementation.

    This class implements the Index protocol and provides all B+Tree
    operations. It uses in-memory nodes for simplicity; a production
    implementation would serialize nodes to pages via the buffer pool.

    Attributes:
        name: Index name.
        table_name: Table being indexed.
        column_name: Column being indexed.
        key_type: Type of indexed keys.
        is_unique: Whether the index enforces uniqueness.
        root_page_id: Page ID of the root node.
        max_keys: Maximum keys per node (fanout - 1).
    """

    name: str
    table_name: str
    column_name: str
    key_type: KeyType
    is_unique: bool = False
    root_page_id: PageId = PageId(0)
    max_keys: int = DEFAULT_MAX_KEYS

    def __post_init__(self) -> None:
        """Initialize the B+Tree with an empty root leaf node."""
        self._lock = threading.RLock()
        self._next_page_id = 1  # Root gets page 0

        # In-memory node storage (simulates buffer pool)
        # Maps page_id -> node
        self._nodes: dict[PageId, BTreeNode] = {}

        # Create root leaf node
        root = BTreeLeafNode.new(PageId(0))
        self._nodes[PageId(0)] = root

        # Statistics
        self._height = 1
        self._num_entries = 0
        self._search_count = 0
        self._insert_count = 0
        self._delete_count = 0

    @property
    def metadata(self) -> IndexMetadata:
        """Return index metadata."""
        return IndexMetadata(
            name=self.name,
            table_name=self.table_name,
            column_name=self.column_name,
            key_type=self.key_type,
            root_page_id=self.root_page_id,
            is_unique=self.is_unique,
            height=self._height,
            num_entries=self._num_entries,
        )

    def _get_node(self, page_id: PageId) -> BTreeNode | None:
        """Get a node from storage."""
        return self._nodes.get(page_id)

    def _allocate_page(self) -> PageId:
        """Allocate a new page ID."""
        page_id = PageId(self._next_page_id)
        self._next_page_id += 1
        return page_id

    def _find_leaf(self, key: IndexKey) -> BTreeLeafNode:
        """Find the leaf node that should contain the given key.

        Traverses from root to leaf, following the appropriate child
        pointers based on key comparisons.

        Args:
            key: The key to search for.

        Returns:
            The leaf node that should contain the key.
        """
        node = self._get_node(self.root_page_id)
        if node is None:
            raise RuntimeError("Root node not found")

        while not node.is_leaf:
            assert isinstance(node, BTreeInternalNode)
            child_page_id = node.find_child(key)
            node = self._get_node(child_page_id)
            if node is None:
                raise RuntimeError(f"Child node {child_page_id} not found")

        assert isinstance(node, BTreeLeafNode)
        return node

    def search(self, key: IndexKey) -> RecordId | None:
        """Search for a key and return its RecordId.

        Args:
            key: The key to search for.

        Returns:
            The RecordId if found, None otherwise.
        """
        with self._lock:
            self._search_count += 1
            leaf = self._find_leaf(key)
            return leaf.search(key)

    def insert(self, key: IndexKey, rid: RecordId) -> bool:
        """Insert a key-value pair into the index.

        For unique indexes, this fails if the key already exists.

        Args:
            key: The key to insert.
            rid: The RecordId to associate with the key.

        Returns:
            True if inserted, False if key already exists (unique index).
        """
        with self._lock:
            self._insert_count += 1

            # Find the leaf node for this key
            leaf = self._find_leaf(key)

            # Check uniqueness for unique indexes
            if self.is_unique and leaf.search(key) is not None:
                return False

            # Insert into leaf
            if not leaf.insert(key, rid):
                return False  # Key already exists

            self._num_entries += 1

            # Check if leaf needs to split
            if leaf.num_keys > self.max_keys:
                self._split_leaf(leaf)

            return True

    def _split_leaf(self, leaf: BTreeLeafNode) -> None:
        """Split a leaf node that has exceeded max_keys.

        Creates a new leaf node and moves half the keys to it.
        Updates parent with new separator key.

        Args:
            leaf: The leaf node to split.
        """
        # Create new leaf node
        new_page_id = self._allocate_page()
        new_leaf = BTreeLeafNode.new(new_page_id)

        # Move half the keys to new leaf
        mid = len(leaf.keys) // 2
        new_leaf.keys = leaf.keys[mid:]
        new_leaf.values = leaf.values[mid:]
        new_leaf.header.num_keys = len(new_leaf.keys)

        leaf.keys = leaf.keys[:mid]
        leaf.values = leaf.values[:mid]
        leaf.header.num_keys = len(leaf.keys)

        # Update sibling pointers
        new_leaf.header.next_page_id = leaf.header.next_page_id
        new_leaf.header.prev_page_id = leaf.page_id
        leaf.header.next_page_id = new_page_id

        if new_leaf.header.next_page_id != INVALID_PAGE_ID:
            next_node = self._get_node(new_leaf.header.next_page_id)
            if next_node and isinstance(next_node, BTreeLeafNode):
                next_node.header.prev_page_id = new_page_id

        # Store new leaf
        self._nodes[new_page_id] = new_leaf

        # Get separator key (first key in new leaf)
        separator = new_leaf.keys[0]

        # Update parent
        self._insert_into_parent(leaf, separator, new_leaf)

    def _insert_into_parent(
        self,
        left_child: BTreeNode,
        key: IndexKey,
        right_child: BTreeNode,
    ) -> None:
        """Insert a separator key into the parent of two children.

        Called after a node split to update the parent with the new
        separator key and child pointer.

        Args:
            left_child: The original child node (already in parent).
            key: The separator key.
            right_child: The new child node (from split).
        """
        parent_page_id = left_child.header.parent_page_id

        if parent_page_id == INVALID_PAGE_ID:
            # Need to create new root
            new_root_page_id = self._allocate_page()
            new_root = BTreeInternalNode.new(new_root_page_id)
            new_root.insert_child(key, left_child.page_id, right_child.page_id)

            # Update children's parent pointers
            left_child.header.parent_page_id = new_root_page_id
            right_child.header.parent_page_id = new_root_page_id

            # Store new root
            self._nodes[new_root_page_id] = new_root
            self.root_page_id = new_root_page_id
            self._height += 1
            return

        # Insert into existing parent
        parent = self._get_node(parent_page_id)
        if parent is None or not isinstance(parent, BTreeInternalNode):
            raise RuntimeError("Invalid parent node")

        parent.insert_child(key, left_child.page_id, right_child.page_id)
        right_child.header.parent_page_id = parent_page_id

        # Check if parent needs to split
        if parent.num_keys > self.max_keys:
            self._split_internal(parent)

    def _split_internal(self, node: BTreeInternalNode) -> None:
        """Split an internal node that has exceeded max_keys.

        Creates a new internal node and moves half the keys to it.
        The middle key moves up to the parent.

        Args:
            node: The internal node to split.
        """
        # Create new internal node
        new_page_id = self._allocate_page()
        new_node = BTreeInternalNode.new(new_page_id)

        # Split keys and children
        mid = len(node.keys) // 2
        separator = node.keys[mid]

        # New node gets keys and children after mid
        new_node.keys = node.keys[mid + 1 :]
        new_node.children = node.children[mid + 1 :]
        new_node.header.num_keys = len(new_node.keys)

        # Original node keeps keys and children before mid
        node.keys = node.keys[:mid]
        node.children = node.children[: mid + 1]
        node.header.num_keys = len(node.keys)

        # Update children's parent pointers
        for child_page_id in new_node.children:
            child = self._get_node(child_page_id)
            if child:
                child.header.parent_page_id = new_page_id

        # Store new node
        self._nodes[new_page_id] = new_node

        # Insert separator into parent
        self._insert_into_parent(node, separator, new_node)

    def delete(self, key: IndexKey) -> bool:
        """Delete a key from the index.

        Note: This is a simplified implementation that doesn't handle
        underflow or node merging. A production implementation would
        rebalance the tree after deletion.

        Args:
            key: The key to delete.

        Returns:
            True if deleted, False if key not found.
        """
        with self._lock:
            self._delete_count += 1

            leaf = self._find_leaf(key)
            if leaf.delete(key):
                self._num_entries -= 1
                return True
            return False

    def update(self, old_key: IndexKey, new_key: IndexKey, rid: RecordId) -> bool:
        """Update a key in the index.

        Equivalent to delete(old_key) + insert(new_key, rid).

        Args:
            old_key: The current key value.
            new_key: The new key value.
            rid: The RecordId.

        Returns:
            True if updated, False if old_key not found.
        """
        with self._lock:
            if not self.delete(old_key):
                return False
            return self.insert(new_key, rid)

    def range_scan(
        self,
        low: IndexKey | None = None,
        high: IndexKey | None = None,
        include_low: bool = True,
        include_high: bool = True,
    ) -> Iterator[tuple[IndexKey, RecordId]]:
        """Scan a range of keys.

        Uses the linked leaf nodes for efficient sequential access.

        Args:
            low: Lower bound (None for unbounded).
            high: Upper bound (None for unbounded).
            include_low: Include the low bound in results.
            include_high: Include the high bound in results.

        Yields:
            (key, rid) tuples in sorted order.
        """
        with self._lock:
            # Find starting leaf
            if low is not None:
                leaf = self._find_leaf(low)
            else:
                # Start from leftmost leaf
                leaf = self._get_leftmost_leaf()

            # Scan through leaves
            while leaf is not None:
                for i, key in enumerate(leaf.keys):
                    # Check lower bound
                    if low is not None:
                        if include_low and key < low:
                            continue
                        if not include_low and key <= low:
                            continue

                    # Check upper bound
                    if high is not None:
                        if include_high and key > high:
                            return
                        if not include_high and key >= high:
                            return

                    yield key, leaf.values[i]

                # Move to next leaf
                if leaf.header.next_page_id == INVALID_PAGE_ID:
                    break
                next_node = self._get_node(leaf.header.next_page_id)
                if not isinstance(next_node, BTreeLeafNode):
                    break
                leaf = next_node

    def _get_leftmost_leaf(self) -> BTreeLeafNode:
        """Find the leftmost leaf node in the tree."""
        node = self._get_node(self.root_page_id)
        if node is None:
            raise RuntimeError("Root node not found")

        while not node.is_leaf:
            assert isinstance(node, BTreeInternalNode)
            if not node.children:
                raise RuntimeError("Internal node has no children")
            child_page_id = node.children[0]
            node = self._get_node(child_page_id)
            if node is None:
                raise RuntimeError(f"Child node {child_page_id} not found")

        assert isinstance(node, BTreeLeafNode)
        return node

    def scan_all(self) -> Iterator[tuple[IndexKey, RecordId]]:
        """Scan all entries in the index.

        Yields:
            (key, rid) tuples in sorted order.
        """
        yield from self.range_scan()


class BTreeIndexManager:
    """Manager for B+Tree indexes.

    Creates, manages, and provides access to indexes. Each index is
    a separate B+Tree instance.

    Thread Safety:
        All methods are thread-safe.
    """

    def __init__(self, buffer_pool: BufferPool | None = None) -> None:
        """Initialize the index manager.

        Args:
            buffer_pool: Optional buffer pool for page access.
                        Currently unused (in-memory implementation).
        """
        self._lock = threading.Lock()
        self._buffer_pool = buffer_pool
        self._indexes: dict[str, BTreeIndex] = {}

        # Statistics
        self._search_count = 0
        self._insert_count = 0
        self._delete_count = 0

    def create_index(
        self,
        name: str,
        table_name: str,
        column_name: str,
        key_type: KeyType,
        is_unique: bool = False,
    ) -> Index:
        """Create a new B+Tree index.

        Args:
            name: The index name.
            table_name: The table being indexed.
            column_name: The column being indexed.
            key_type: The type of the indexed column.
            is_unique: Whether the index enforces uniqueness.

        Returns:
            The created index.

        Raises:
            ValueError: If index name already exists.
        """
        with self._lock:
            if name in self._indexes:
                raise ValueError(f"Index '{name}' already exists")

            index = BTreeIndex(
                name=name,
                table_name=table_name,
                column_name=column_name,
                key_type=key_type,
                is_unique=is_unique,
            )
            self._indexes[name] = index
            return index

    def get_index(self, name: str) -> Index | None:
        """Get an index by name.

        Args:
            name: The index name.

        Returns:
            The index if found, None otherwise.
        """
        with self._lock:
            return self._indexes.get(name)

    def drop_index(self, name: str) -> bool:
        """Drop an index.

        Args:
            name: The index name.

        Returns:
            True if dropped, False if not found.
        """
        with self._lock:
            if name in self._indexes:
                del self._indexes[name]
                return True
            return False

    def list_indexes(self, table_name: str | None = None) -> list[IndexMetadata]:
        """List all indexes, optionally filtered by table.

        Args:
            table_name: If provided, only return indexes for this table.

        Returns:
            List of index metadata.
        """
        with self._lock:
            indexes = self._indexes.values()
            if table_name is not None:
                indexes = [idx for idx in indexes if idx.table_name == table_name]
            return [idx.metadata for idx in indexes]

    def get_stats(self) -> IndexStats:
        """Return index manager statistics for monitoring."""
        with self._lock:
            total_entries = sum(idx._num_entries for idx in self._indexes.values())
            heights = [idx._height for idx in self._indexes.values()]
            avg_height = sum(heights) / len(heights) if heights else 0.0
            search_count = sum(idx._search_count for idx in self._indexes.values())
            insert_count = sum(idx._insert_count for idx in self._indexes.values())
            delete_count = sum(idx._delete_count for idx in self._indexes.values())

            return IndexStats(
                num_indexes=len(self._indexes),
                total_entries=total_entries,
                avg_height=avg_height,
                search_count=search_count,
                insert_count=insert_count,
                delete_count=delete_count,
            )
