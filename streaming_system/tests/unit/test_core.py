"""Unit tests for streaming_system domain layer."""

import pytest
from streaming_system.domain.entities.record import (
    Record,
    LogEntry,
    LogSegment,
    HighWaterMark,
    RecordCompressionType,
)
from streaming_system.domain.entities.raft import (
    RaftTerm,
    RaftStateEnum,
    InSyncReplicas,
    RaftNode,
)
from streaming_system.domain.entities.consumer import (
    ConsumerMember,
    ConsumerGroup,
    ConsumerOffset,
    PartitionAssignment,
    ConsumerState,
)
from streaming_system.domain.services.partition_log import PartitionLog
from streaming_system.domain.services.raft_consensus import RaftConsensus
from streaming_system.domain.services.consumer_coordinator import ConsumerCoordinator
from streaming_system.domain.value_objects.identifiers import (
    create_node_id,
    create_consumer_id,
    create_consumer_group_id,
)


class TestIdentifiers:
    """Test value object creation."""
    
    def test_create_node_id(self):
        """Test node ID creation."""
        node_id = create_node_id("broker-0")
        assert "node-" in node_id
        assert "broker-0" in node_id
    
    def test_create_consumer_id(self):
        """Test consumer ID creation."""
        consumer_id = create_consumer_id("my-group", 0)
        assert "my-group" in consumer_id
        assert "consumer" in consumer_id
    
    def test_create_consumer_group_id(self):
        """Test consumer group ID creation."""
        group_id = create_consumer_group_id("readers")
        assert "group-" in group_id
        assert "readers" in group_id


class TestRecord:
    """Test record entities."""
    
    def test_record_creation(self):
        """Test creating a record."""
        record = Record(
            key=b"key1",
            value=b"value1",
        )
        assert record.key == b"key1"
        assert record.value == b"value1"
    
    def test_record_crc(self):
        """Test CRC computation."""
        record = Record(key=b"key", value=b"value")
        crc1 = record.compute_crc()
        crc2 = record.compute_crc()
        
        assert crc1 == crc2  # Deterministic
        assert len(crc1) == 8  # Hex string


class TestHighWaterMark:
    """Test high water mark tracking."""
    
    def test_hwm_advance(self):
        """Test advancing HWM."""
        hwm = HighWaterMark()
        assert hwm.get() == 0
        
        hwm.advance(5)
        assert hwm.get() == 5
        
        # Can't go backwards
        hwm.advance(3)
        assert hwm.get() == 5


class TestRaftTerm:
    """Test Raft term management."""
    
    def test_term_increment(self):
        """Test incrementing term."""
        term = RaftTerm(value=5)
        term.increment()
        assert term.get() == 6
    
    def test_term_update(self):
        """Test updating term."""
        term = RaftTerm(value=5)
        
        # Update to higher term
        assert term.update(7)
        assert term.get() == 7
        
        # Can't go backwards
        assert not term.update(6)
        assert term.get() == 7


class TestInSyncReplicas:
    """Test ISR tracking."""
    
    def test_isr_membership(self):
        """Test adding/removing from ISR."""
        isr = InSyncReplicas()
        
        isr.add("node-0")
        isr.add("node-1")
        
        assert isr.contains("node-0")
        assert isr.contains("node-1")
        assert not isr.contains("node-2")
        
        isr.remove("node-0")
        assert not isr.contains("node-0")
    
    def test_quorum_check(self):
        """Test quorum calculation."""
        isr = InSyncReplicas(members={"node-0", "node-1", "node-2"})
        
        # Need 2 out of 3 for quorum
        assert isr.is_quorum(2)
        assert not isr.is_quorum(1)


class TestPartitionLog:
    """Test partition log service."""
    
    def test_append_and_read(self):
        """Test appending and reading records."""
        log = PartitionLog(partition_id=0)
        
        records = [
            Record(key=b"k1", value=b"v1"),
            Record(key=b"k2", value=b"v2"),
            Record(key=b"k3", value=b"v3"),
        ]
        
        offsets = log.append(records)
        
        assert len(offsets) == 3
        assert offsets == [0, 1, 2]
        assert log.get_next_offset() == 3
    
    def test_commit(self):
        """Test committing entries."""
        log = PartitionLog(partition_id=0)
        
        records = [Record(key=b"k", value=b"v") for _ in range(5)]
        log.append(records)
        
        log.commit(2)
        assert log.get_hwm() == 2
        
        # Entries should be marked committed
        entries = log.read(0, 5)
        assert entries[0].is_committed
        assert entries[1].is_committed
        assert entries[2].is_committed
        assert not entries[3].is_committed
    
    def test_truncate(self):
        """Test log truncation."""
        log = PartitionLog(partition_id=0)
        
        records = [Record(key=b"k", value=b"v") for _ in range(5)]
        log.append(records)
        
        log.truncate(3)
        
        assert log.get_next_offset() == 3
        assert len(log.read(0, 10)) == 3


class TestRaftConsensus:
    """Test Raft consensus."""
    
    def test_follower_grants_vote(self):
        """Test follower granting vote."""
        raft = RaftConsensus(node_id="node-0", cluster_size=3)

        granted = raft.request_vote(
            term=1,
            candidate_id="node-1",
            last_log_index=0,
            last_log_term=0,
        )

        assert granted
        assert raft.node.voted_for == "node-1"

    def test_reject_old_term(self):
        """Test rejecting vote from old term."""
        raft = RaftConsensus(node_id="node-0", cluster_size=3)
        raft.node.current_term.value = 5

        # Old term should be rejected
        granted = raft.request_vote(
            term=3,
            candidate_id="node-1",
            last_log_index=0,
            last_log_term=0,
        )

        assert not granted

    def test_reject_stale_log(self):
        """Test rejecting vote from candidate with stale log."""
        raft = RaftConsensus(node_id="node-0", cluster_size=3)

        # Add some entries to our log
        raft.append_log_entry(term=1, data=b"entry1")
        raft.append_log_entry(term=2, data=b"entry2")

        # Candidate with older log term should be rejected
        granted = raft.request_vote(
            term=3,
            candidate_id="node-1",
            last_log_index=1,
            last_log_term=1,  # Our last term is 2
        )

        assert not granted

    def test_append_entries_updates_leader(self):
        """Test append entries from leader."""
        raft = RaftConsensus(node_id="node-0", cluster_size=3)

        result = raft.append_entries(
            term=1,
            leader_id="node-1",
            prev_log_index=0,
            prev_log_term=0,
            entries=[],
        )

        assert result
        assert raft.node.leader_id == "node-1"

    def test_append_entries_log_consistency(self):
        """Test log consistency check in append entries."""
        from streaming_system.domain.services.raft_consensus import LogEntry

        raft = RaftConsensus(node_id="node-0", cluster_size=3)

        # Add initial entry
        raft.append_log_entry(term=1, data=b"entry1")

        # Append with correct prev_log should succeed
        result = raft.append_entries(
            term=1,
            leader_id="node-1",
            prev_log_index=1,
            prev_log_term=1,
            entries=[LogEntry(index=2, term=1, data=b"entry2")],
        )
        assert result
        assert raft.get_last_log_index() == 2

        # Append with wrong prev_log_term should fail
        result = raft.append_entries(
            term=1,
            leader_id="node-1",
            prev_log_index=2,
            prev_log_term=99,  # Wrong term
            entries=[LogEntry(index=3, term=1, data=b"entry3")],
        )
        assert not result
    
    def test_election_timeout(self):
        """Test election timeout."""
        raft = RaftConsensus(node_id="node-0", cluster_size=3)
        
        # Initially no timeout
        assert not raft.check_election_timeout()
        
        # Simulate timeout by setting old heartbeat
        raft._last_heartbeat = 0
        assert raft.check_election_timeout()


class TestConsumerGroup:
    """Test consumer group management."""
    
    def test_add_member(self):
        """Test adding member to group."""
        group = ConsumerGroup(group_id="group-1")
        
        member = ConsumerMember(
            consumer_id="consumer-1",
            group_id="group-1",
        )
        
        group.add_member(member)
        assert "consumer-1" in group.members
    
    def test_rebalance(self):
        """Test rebalancing."""
        group = ConsumerGroup(group_id="group-1")
        
        for i in range(3):
            member = ConsumerMember(
                consumer_id=f"consumer-{i}",
                group_id="group-1",
            )
            group.add_member(member)
        
        group.start_rebalance()
        assert group.state.value == "rebalancing"
        assert group.generation == 1
        
        group.complete_rebalance()
        assert group.state.value == "stable"


class TestConsumerCoordinator:
    """Test consumer coordinator service."""
    
    def test_create_group(self):
        """Test creating consumer group."""
        coordinator = ConsumerCoordinator()
        
        group = coordinator.create_group("my-group")
        assert group.group_id == "my-group"
        assert coordinator.get_group("my-group") is not None
    
    def test_join_group(self):
        """Test consumer joining group."""
        coordinator = ConsumerCoordinator()
        
        group = coordinator.join_group("my-group", "consumer-1")
        
        assert "consumer-1" in group.members
        assert group.state.value == "rebalancing"
    
    def test_offset_management(self):
        """Test offset commit and fetch."""
        coordinator = ConsumerCoordinator()
        
        coordinator.commit_offset("my-group", 0, 100)
        offset = coordinator.get_offset("my-group", 0)
        
        assert offset == 100
    
    def test_partition_assignment(self):
        """Test partition assignment (range strategy)."""
        coordinator = ConsumerCoordinator()
        
        # Create group with 3 consumers
        for i in range(3):
            coordinator.join_group("my-group", f"consumer-{i}")
        
        # Assign 6 partitions
        assignment = coordinator.assign_partitions("my-group", 6)
        
        assert len(assignment) == 3
        assert assignment["consumer-0"] == [0, 1]
        assert assignment["consumer-1"] == [2, 3]
        assert assignment["consumer-2"] == [4, 5]
    
    def test_heartbeat(self):
        """Test consumer heartbeat."""
        coordinator = ConsumerCoordinator()
        coordinator.join_group("my-group", "consumer-1")
        
        assert coordinator.heartbeat("my-group", "consumer-1")
        assert not coordinator.heartbeat("my-group", "consumer-2")
