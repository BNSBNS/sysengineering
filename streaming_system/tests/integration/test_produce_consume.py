"""Integration tests for streaming system produce/consume workflow."""

import pytest
import time

from streaming_system.domain.entities.record import Record, LogEntry
from streaming_system.domain.entities.consumer import ConsumerGroup, ConsumerMember
from streaming_system.domain.services.partition_log import PartitionLog
from streaming_system.domain.services.consumer_coordinator import ConsumerCoordinator
from streaming_system.domain.services.raft_consensus import RaftNode, RaftState


class TestPartitionLogIntegration:
    """Integration tests for partition log operations."""

    def test_append_and_read(self):
        """Test appending and reading records from partition log."""
        log = PartitionLog(partition_id=0)

        # Append records
        records = [
            Record(key=b"key1", value=b"value1"),
            Record(key=b"key2", value=b"value2"),
            Record(key=b"key3", value=b"value3"),
        ]
        offsets = log.append(records)

        assert len(offsets) == 3
        assert offsets == [0, 1, 2]

        # Read back
        entries = log.read(offset=0, max_records=10)
        assert len(entries) == 3

        for i, entry in enumerate(entries):
            assert entry.offset == i
            assert entry.record.key == records[i].key
            assert entry.record.value == records[i].value

    def test_commit_advances_hwm(self):
        """Test that committing advances high water mark."""
        log = PartitionLog(partition_id=0)

        # Append records
        records = [Record(key=b"k", value=b"v") for _ in range(5)]
        log.append(records)

        # Initial HWM is 0
        assert log.get_hwm() == 0

        # Commit up to offset 2
        log.commit(2)
        assert log.get_hwm() == 2

        # Verify entries are marked committed
        entries = log.read(0, 10)
        for entry in entries[:3]:  # 0, 1, 2
            assert entry.is_committed
        for entry in entries[3:]:  # 3, 4
            assert not entry.is_committed

    def test_truncate_for_raft_recovery(self):
        """Test log truncation for Raft recovery scenarios."""
        log = PartitionLog(partition_id=0)

        # Append 5 records
        records = [Record(key=f"k{i}".encode(), value=b"v") for i in range(5)]
        log.append(records)

        assert log.get_next_offset() == 5

        # Truncate at offset 3 (remove entries 3 and 4)
        log.truncate(3)

        assert log.get_next_offset() == 3
        entries = log.read(0, 10)
        assert len(entries) == 3

    def test_raft_term_tracking(self):
        """Test that Raft term is tracked in log entries."""
        log = PartitionLog(partition_id=0)

        # Term 1
        log.set_term(1)
        log.append([Record(key=b"k1", value=b"v1")])

        # Term 2
        log.set_term(2)
        log.append([Record(key=b"k2", value=b"v2")])

        entries = log.read(0, 10)
        assert entries[0].term == 1
        assert entries[1].term == 2


class TestConsumerCoordinatorIntegration:
    """Integration tests for consumer group coordination."""

    def test_consumer_group_lifecycle(self):
        """Test consumer group creation, join, and leave."""
        coordinator = ConsumerCoordinator()

        # Create group
        group = coordinator.create_group("test-group")
        assert group.group_id == "test-group"

        # Join consumers
        coordinator.join_group("test-group", "consumer-1")
        coordinator.join_group("test-group", "consumer-2")

        group = coordinator.get_group("test-group")
        assert group.size() == 2

        # Leave
        coordinator.leave_group("test-group", "consumer-1")
        group = coordinator.get_group("test-group")
        assert group.size() == 1

    def test_partition_assignment(self):
        """Test partition assignment to consumers."""
        coordinator = ConsumerCoordinator()

        # Create group and join consumers
        coordinator.create_group("my-group")
        coordinator.join_group("my-group", "c1")
        coordinator.join_group("my-group", "c2")

        # Assign 6 partitions to 2 consumers
        assignment = coordinator.assign_partitions("my-group", partition_count=6)

        assert len(assignment) == 2
        assert len(assignment["c1"]) == 3
        assert len(assignment["c2"]) == 3

        # Check consecutive assignment (range strategy)
        assert assignment["c1"] == [0, 1, 2]
        assert assignment["c2"] == [3, 4, 5]

    def test_offset_management(self):
        """Test consumer offset commit and retrieval."""
        coordinator = ConsumerCoordinator()

        group_id = "offset-test-group"
        coordinator.create_group(group_id)
        coordinator.join_group(group_id, "consumer-1")

        # Initial offset is 0
        assert coordinator.get_offset(group_id, partition_id=0) == 0

        # Commit offset
        coordinator.commit_offset(group_id, partition_id=0, offset=100)
        assert coordinator.get_offset(group_id, partition_id=0) == 100

        # Commit higher offset
        coordinator.commit_offset(group_id, partition_id=0, offset=200)
        assert coordinator.get_offset(group_id, partition_id=0) == 200

    def test_heartbeat_session(self):
        """Test consumer heartbeat for session management."""
        coordinator = ConsumerCoordinator()

        coordinator.create_group("heartbeat-group")
        coordinator.join_group("heartbeat-group", "consumer-1")

        # Heartbeat should succeed for joined consumer
        result = coordinator.heartbeat("heartbeat-group", "consumer-1")
        assert result

        # Heartbeat should fail for unknown consumer
        result = coordinator.heartbeat("heartbeat-group", "unknown-consumer")
        assert not result


class TestProduceConsumeWorkflow:
    """Integration tests for complete produce/consume workflow."""

    def test_end_to_end_workflow(self):
        """Test complete produce -> commit -> consume workflow."""
        # Setup
        log = PartitionLog(partition_id=0)
        coordinator = ConsumerCoordinator()

        # Create consumer group
        coordinator.create_group("e2e-group")
        coordinator.join_group("e2e-group", "consumer-1")
        coordinator.assign_partitions("e2e-group", partition_count=1)

        # Produce messages
        messages = [
            Record(key=b"user:1", value=b'{"action": "login"}'),
            Record(key=b"user:2", value=b'{"action": "purchase"}'),
            Record(key=b"user:1", value=b'{"action": "logout"}'),
        ]
        offsets = log.append(messages)

        # Commit (simulate Raft replication)
        log.commit(offsets[-1])

        # Consume from start
        consumer_offset = coordinator.get_offset("e2e-group", partition_id=0)
        entries = log.read(offset=consumer_offset, max_records=100)

        assert len(entries) == 3
        for entry in entries:
            assert entry.is_committed

        # Commit consumer offset
        last_offset = entries[-1].offset
        coordinator.commit_offset("e2e-group", partition_id=0, offset=last_offset)

        # Verify consumer position
        assert coordinator.get_offset("e2e-group", partition_id=0) == last_offset

    def test_multi_partition_consumption(self):
        """Test consuming from multiple partitions."""
        # Setup multiple partition logs
        partitions = [PartitionLog(partition_id=i) for i in range(3)]
        coordinator = ConsumerCoordinator()

        # Create consumer group with 2 consumers
        coordinator.create_group("multi-partition-group")
        coordinator.join_group("multi-partition-group", "c1")
        coordinator.join_group("multi-partition-group", "c2")
        assignment = coordinator.assign_partitions("multi-partition-group", partition_count=3)

        # Produce to each partition
        for i, log in enumerate(partitions):
            records = [Record(key=f"p{i}".encode(), value=f"msg{j}".encode()) for j in range(5)]
            log.append(records)
            log.commit(4)

        # Each consumer reads from assigned partitions
        for consumer_id, assigned in assignment.items():
            total_messages = 0
            for partition_id in assigned:
                offset = coordinator.get_offset("multi-partition-group", partition_id)
                entries = partitions[partition_id].read(offset, 100)
                total_messages += len(entries)

            # Each partition has 5 messages
            assert total_messages == len(assigned) * 5


class TestRaftConsensusIntegration:
    """Integration tests for Raft consensus with partition log."""

    def test_raft_node_initialization(self):
        """Test Raft node initialization."""
        node = RaftNode(node_id="node-1", cluster_size=3)

        assert node.state == RaftState.FOLLOWER
        assert node.current_term == 0
        assert node.voted_for is None

    def test_leader_election(self):
        """Test Raft leader election process."""
        # Create cluster of 3 nodes
        nodes = [RaftNode(node_id=f"node-{i}", cluster_size=3) for i in range(3)]

        # Node 0 starts election
        nodes[0].start_election()
        assert nodes[0].state == RaftState.CANDIDATE
        assert nodes[0].current_term == 1

        # Other nodes vote
        vote1 = nodes[1].request_vote(
            candidate_id="node-0",
            term=1,
            last_log_index=0,
            last_log_term=0,
        )
        vote2 = nodes[2].request_vote(
            candidate_id="node-0",
            term=1,
            last_log_index=0,
            last_log_term=0,
        )

        assert vote1  # Vote granted
        assert vote2  # Vote granted

        # Node 0 wins election
        nodes[0].win_election()
        assert nodes[0].state == RaftState.LEADER

    def test_append_entries_replication(self):
        """Test log replication via append entries."""
        # Create leader and follower
        leader = RaftNode(node_id="leader", cluster_size=3)
        follower = RaftNode(node_id="follower", cluster_size=3)

        # Leader starts in term 1
        leader.start_election()
        leader.win_election()

        # Follower accepts entries from leader
        success, match_index = follower.append_entries(
            leader_id="leader",
            term=1,
            prev_log_index=0,
            prev_log_term=0,
            entries=[(1, b"data1"), (1, b"data2")],
            leader_commit=0,
        )

        assert success
        assert match_index == 2

        # Follower's log should have entries
        assert follower.get_last_log_index() == 2

    def test_integrated_raft_with_partition_log(self):
        """Test Raft consensus integrated with partition log."""
        log = PartitionLog(partition_id=0)
        node = RaftNode(node_id="node-1", cluster_size=3)

        # Become leader
        node.start_election()
        node.win_election()

        # Set log term
        log.set_term(node.current_term)

        # Append entries
        records = [Record(key=b"k", value=b"v")]
        offsets = log.append(records)

        # After replication, commit
        log.commit(offsets[0])

        # Verify integration
        stats = log.get_stats()
        assert stats["current_term"] == 1
        assert stats["committed_offset"] == 0


class TestLogSegmentation:
    """Integration tests for log segment management."""

    def test_segment_rollover(self):
        """Test that segments roll over when full."""
        log = PartitionLog(partition_id=0)

        # Create small segment for testing
        log._segments[-1].max_segment_bytes = 100  # Very small for testing

        # Append records until segment rolls
        initial_segments = len(log._segments)

        for i in range(50):
            log.append([Record(key=f"key{i}".encode(), value=b"x" * 10)])

        # Should have created new segments
        assert len(log._segments) > initial_segments


class TestConsumerLag:
    """Integration tests for consumer lag tracking."""

    def test_lag_tracking(self):
        """Test consumer lag calculation."""
        log = PartitionLog(partition_id=0)
        coordinator = ConsumerCoordinator()

        # Setup
        coordinator.create_group("lag-group")
        coordinator.join_group("lag-group", "consumer-1")
        coordinator.assign_partitions("lag-group", partition_count=1)

        # Produce 100 messages
        records = [Record(key=b"k", value=b"v") for _ in range(100)]
        log.append(records)
        log.commit(99)  # All committed

        # Consumer at offset 50
        coordinator.commit_offset("lag-group", partition_id=0, offset=50)

        # Calculate lag
        consumer_offset = coordinator.get_offset("lag-group", partition_id=0)
        hwm = log.get_hwm()
        lag = hwm - consumer_offset

        assert lag == 49  # 99 - 50 = 49 messages behind


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
