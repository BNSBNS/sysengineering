# Distributed Streaming System Design Document

## High-Level Component Overview

This document describes a replicated, partitioned log system implementing the principles from Apache Kafka's design, the Raft consensus protocol from Stanford, and Jay Kreps' seminal work "The Log: What every software engineer should know."

### Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Streaming System                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Producers  │───▶│    Broker    │───▶│   Consumers  │                   │
│  │   (Writers)  │    │  (Partition) │    │   (Groups)   │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    Consensus & Replication                        │       │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │       │
│  │  │    Raft    │  │    ISR     │  │   Leader   │  │  Failover  │  │       │
│  │  │  Consensus │  │  Tracking  │  │  Election  │  │  Manager   │  │       │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Each Component Exists

| Component | Purpose | Why This Approach | Source |
|-----------|---------|-------------------|--------|
| **Partitioned Log** | Horizontal scaling | Single log can't scale beyond one disk. Partitions enable parallelism [1] | Kreps "The Log" |
| **Raft Consensus** | Leader election | Understandable consensus (vs Paxos). Proven correct [2] | Ongaro & Ousterhout 2014 |
| **ISR (In-Sync Replicas)** | Durability + availability | Only replicas caught up can become leader. Balances durability/availability [3] | Kafka design |
| **Consumer Groups** | Parallel consumption | Multiple consumers share partitions. Auto-rebalance on failure [4] | Kafka protocol |
| **Segmented Storage** | Efficient retention | Old segments deleted without rewriting. O(1) append [5] | Log-structured storage |
| **Offset Tracking** | Exactly-once (with idempotence) | Consumers track position. Resume after crash [6] | Kafka consumer protocol |

### Why "The Log" Is Fundamental

```
The Log as Universal Data Structure:

Database WAL:           Message Queue:           Event Sourcing:
┌─────────────┐        ┌─────────────┐          ┌─────────────┐
│  Append     │        │   Publish   │          │   Record    │
│  Record     │        │   Message   │          │   Event     │
└──────┬──────┘        └──────┬──────┘          └──────┬──────┘
       │                      │                        │
       ▼                      ▼                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   Append-Only Log                            │
│  [offset 0] [offset 1] [offset 2] [offset 3] [offset 4] ... │
└─────────────────────────────────────────────────────────────┘

Properties:
- Total ordering (offset = position)
- Immutable (append-only)
- Replayable (read from any offset)
- Durable (fsync guarantees)
```

### References

1. **The Log**: Kreps, J. "The Log: What every software engineer should know about real-time data's unifying abstraction" LinkedIn Engineering (2013)
2. **Raft**: Ongaro, D. & Ousterhout, J. "In Search of an Understandable Consensus Algorithm" USENIX ATC (2014)
3. **Kafka Design**: kafka.apache.org/documentation/#design
4. **Consumer Groups**: Kafka Improvement Proposal KIP-429 "Consumer Group Protocol"
5. **Log-Structured Storage**: Rosenblum, M. & Ousterhout, J. "The Design and Implementation of a Log-Structured File System" SOSP (1991)
6. **Exactly-Once**: Kafka Improvement Proposal KIP-98 "Exactly Once Delivery and Transactional Messaging"

---

## 1. Problem Statement & Non-Goals

### Problem Statement

Build a replicated, partitioned log system with Raft consensus, durability guarantees, and consumer group semantics.

**Why build this?** Understanding distributed logs is essential for:
- Building event-driven architectures (microservices communication)
- Implementing CQRS/Event Sourcing patterns
- Understanding database replication internals
- Designing real-time data pipelines

### Goals

- Partitioned append-only log with segmented storage
- Raft consensus for leader election and replication
- In-Sync Replicas (ISR) tracking for durability
- Consumer groups with automatic rebalancing
- No data loss after acknowledged writes

### Non-Goals

- Exactly-once semantics (at-least-once only)
- Schema registry
- Stream processing (Kafka Streams equivalent)
- Multi-datacenter replication

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client Layer                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │  Producers  │  │  Consumers  │  │    Admin API                │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                         Broker Layer                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │ Partition   │  │   Raft      │  │    Consumer                 │  │
│  │   Log       │  │  Consensus  │  │    Coordinator              │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                       Replication Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │    ISR      │  │  Controller │  │    Offset Manager           │  │
│  │  Tracking   │  │             │  │                             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                        Storage Layer                                 │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │              Segmented Log (append-only, indexed)               ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### How Components Interact

1. **Produce Flow**: Producer → Leader broker → Append to log → Replicate to ISR → Ack
2. **Consume Flow**: Consumer → Coordinator → Fetch from partition → Track offset
3. **Replication Flow**: Leader → Append entries → Followers ack → Commit → Advance high watermark

---

## 3. Core Components & APIs

### Partition Log

**Why append-only?**
```
Read-modify-write:              Append-only:
┌─────────────────┐            ┌─────────────────┐
│ Read record     │            │ Append record   │
│ Modify in place │            │ (O(1), no locks)│
│ Write back      │            └─────────────────┘
│ (random I/O)    │
└─────────────────┘            Sequential I/O = 100x faster than random
                               SSD: 500 MB/s sequential vs 5 MB/s random
```

```python
class PartitionLog(Protocol):
    def append(self, records: list[Record]) -> list[Offset]: ...
    def read(self, offset: Offset, max_records: int) -> list[Record]: ...
    def truncate(self, offset: Offset) -> None: ...
    def get_high_watermark(self) -> Offset: ...
```

### Raft Consensus

**Why Raft over Paxos?**
```
Paxos:                          Raft:
- Optimal messages              - Slightly more messages
- Very difficult to understand  - Designed for understandability
- Many variants, no standard    - Single specification
- "Paxos is too hard"           - Used by etcd, CockroachDB, TiKV
  - Leslie Lamport
```

**Raft state machine**:
```
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ▼                                         │
             ┌─────────────┐  election timeout   ┌─────────────┐
             │  FOLLOWER   │ ─────────────────▶  │  CANDIDATE  │
             └─────────────┘                     └─────────────┘
                    ▲                                   │
                    │                              win election
                    │                                   │
                    │                                   ▼
                    │                            ┌─────────────┐
                    └────────────────────────────│   LEADER    │
                             lose leadership     └─────────────┘
                             (higher term seen)
```

```python
class RaftNode(Protocol):
    def request_vote(self, term: int, candidate: NodeId) -> VoteResponse: ...
    def append_entries(self, entries: list[LogEntry]) -> AppendResponse: ...
    def get_leader(self) -> NodeId | None: ...
```

### Consumer Coordinator

**Why consumer groups?**
```
Without groups:                 With groups:
Consumer 1 → All partitions    Consumer 1 → Partitions [0, 1]
Consumer 2 → All partitions    Consumer 2 → Partitions [2, 3]
Consumer 3 → All partitions    Consumer 3 → Partitions [4, 5]

Result: 3x duplicate           Result: Parallel processing
        processing                     with load balancing
```

```python
class ConsumerCoordinator(Protocol):
    def join_group(self, group: str, consumer: ConsumerId) -> Assignment: ...
    def leave_group(self, group: str, consumer: ConsumerId) -> None: ...
    def commit_offset(self, group: str, partition: int, offset: Offset) -> None: ...
    def fetch_offset(self, group: str, partition: int) -> Offset: ...
```

---

## 4. Data Models & State Machines

### Raft State Machine

```
Term: Logical clock that increases on each election
      Prevents stale leaders from causing inconsistency

Leader Election:
1. Follower times out (no heartbeat from leader)
2. Becomes Candidate, increments term, votes for self
3. Requests votes from all nodes
4. If majority votes received → becomes Leader
5. If higher term seen → becomes Follower

Log Replication:
1. Leader receives write from client
2. Appends to local log (uncommitted)
3. Sends AppendEntries to all followers
4. When majority ack → entry is committed
5. Leader applies to state machine, responds to client
```

### Log Segment Structure

**Why segments?**
```
Single file:                    Segmented:
┌─────────────────────────┐    ┌──────────┐  ┌──────────┐  ┌──────────┐
│ All records since       │    │ Segment 0│  │ Segment 1│  │ Segment 2│
│ beginning of time       │    │ (closed) │  │ (closed) │  │ (active) │
│ (can't delete old data) │    │ delete ✓ │  │ delete ✓ │  │ append ✓ │
└─────────────────────────┘    └──────────┘  └──────────┘  └──────────┘

Retention: Delete entire segments, no rewriting
           Typical: 1GB segments, 7 days retention
```

```
Segment File: 00000000000012345678.log
├── Record 0: [offset=12345678, timestamp, key, value, crc]
├── Record 1: [offset=12345679, timestamp, key, value, crc]
└── ...

Index File: 00000000000012345678.index
├── [relative_offset=0, position=0]
├── [relative_offset=100, position=8192]
└── ...  (sparse index, every N records)

Time Index: 00000000000012345678.timeindex
├── [timestamp=1704067200, offset=0]
├── [timestamp=1704067260, offset=500]
└── ...  (enables time-based lookup)
```

---

## 5. Concurrency Model

| Component | Model | Rationale |
|-----------|-------|-----------|
| Network I/O | asyncio | High concurrency connections |
| Disk I/O | aiofiles | Non-blocking append/read |
| Raft heartbeats | Threading | Timing critical (150-300ms) |
| Compaction | multiprocessing | CPU-bound, background |

### Producer Batching

```python
class BatchingProducer:
    """
    Batch records for efficiency.

    Without batching: 1 record = 1 network call = 1 disk write
    With batching: 1000 records = 1 network call = 1 disk write

    Throughput improvement: 100-1000x
    """

    async def send(self, topic: str, key: bytes, value: bytes):
        batch = self.get_or_create_batch(topic, partition)
        batch.append(Record(key, value))

        if batch.is_full() or batch.is_expired():
            await self.flush_batch(batch)
```

---

## 6. Failure Modes & Recovery

| Failure | Detection | Recovery | Source |
|---------|-----------|----------|--------|
| Leader crash | Election timeout | Raft election (new leader) | Raft paper |
| Network partition | Heartbeat failure | Partition healing, ISR update | Kafka ISR |
| Disk corruption | CRC mismatch | Replicate from replica | Checksums |
| Consumer crash | Session timeout | Rebalance partitions | Consumer protocol |
| Slow follower | Replica lag metric | Remove from ISR | Kafka ISR |

### ISR (In-Sync Replicas) Mechanism

```
ISR balances durability vs availability:

All replicas:     [Broker 1 (Leader), Broker 2, Broker 3]
ISR (caught up):  [Broker 1, Broker 2]  (Broker 3 is lagging)

Write with acks=all:
1. Leader receives write
2. Waits for all ISR members to ack
3. Does NOT wait for Broker 3 (not in ISR)
4. Returns success to client

If Leader fails:
- Only ISR members can become new leader
- Guarantees no data loss for committed records
- Broker 3 cannot become leader (would lose data)

Trade-off:
- ISR size = 1: Maximum availability, risk of data loss
- ISR size = all replicas: Maximum durability, lower availability
- Default: min.insync.replicas = 2 (balance)
```

---

## 7. Security Threat Model

| Threat | Mitigation | Implementation |
|--------|------------|----------------|
| Unauthorized access | ACLs, SASL authentication | Per-topic, per-consumer-group ACLs |
| Data tampering | CRC checksums per record | Verify on read, reject corrupted |
| Replay attacks | Offset monotonicity | Offsets only increase |
| Eavesdropping | TLS encryption | In-flight encryption |
| DoS | Rate limiting, quotas | Per-client byte/request limits |

### Authentication Flow

```
SASL/SCRAM Authentication:
1. Client → Server: SASL handshake (mechanism: SCRAM-SHA-256)
2. Server → Client: Challenge (server nonce)
3. Client → Server: Response (client proof)
4. Server → Client: Success (server signature)

After auth: All requests include authenticated principal
           ACL checks: principal + resource + operation → allow/deny
```

---

## 8. Performance Targets

| Metric | Target | How Achieved |
|--------|--------|--------------|
| Produce latency (ack=leader) | < 5ms p99 | Batching, sequential I/O |
| Produce latency (ack=all) | < 20ms p99 | Async replication, ISR optimization |
| Consume throughput | > 100 MB/s | Zero-copy sendfile(), batching |
| Leader election | < 10 seconds | Raft election timeout tuning |
| Rebalance time | < 30 seconds | Incremental cooperative rebalancing |

### Latency Breakdown

```
Produce (ack=all) latency:
├── Network client→leader:     2ms
├── Append to leader log:      1ms (sequential write)
├── Replicate to followers:    5ms (parallel, wait for ISR)
├── Commit (advance HW):       0ms (in-memory)
├── Network leader→client:     2ms
└── Total:                    ~10ms typical

Zero-copy optimization (consume):
Traditional:                   Zero-copy (sendfile):
1. Disk → Kernel buffer       1. Disk → Kernel buffer
2. Kernel → User buffer       2. Kernel → Socket buffer (DMA)
3. User → Socket buffer
4. Socket → Network           3. Socket → Network

Copies: 4 → 2                 CPU usage: -50%
```

---

## 9. Operational Concerns

### Topic Management

```bash
# Create topic with partitions and replication
streaming-system topics create --name events --partitions 3 --replication-factor 3

# Describe topic (shows partition leaders, ISR)
streaming-system topics describe --name events
# Output:
# Partition 0: Leader=1, ISR=[1,2,3]
# Partition 1: Leader=2, ISR=[2,3,1]
# Partition 2: Leader=3, ISR=[3,1,2]

# Increase partitions (cannot decrease)
streaming-system topics alter --name events --partitions 6
```

### Consumer Groups

```bash
# List all consumer groups
streaming-system groups list

# Describe group (shows members, partition assignments, lag)
streaming-system groups describe --group my-group
# Output:
# Consumer-1: Partitions [0, 1], Lag: 100
# Consumer-2: Partitions [2], Lag: 0

# Reset offsets (for reprocessing)
streaming-system groups reset-offsets --group my-group --topic events --to-earliest
```

### Monitoring

```bash
# Check cluster health
streaming-system cluster status

# Show under-replicated partitions
streaming-system cluster under-replicated

# Show consumer lag by group
streaming-system lag --group my-group
```

---

## 10. Alternatives Considered

### Consensus Protocol

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Raft** | Understandable, proven, single spec | Slightly more messages | **Selected** |
| Paxos | Optimal message complexity | Very complex, many variants | Rejected |
| ZAB (ZooKeeper) | Battle-tested in Kafka | Requires separate ZK cluster | Rejected |
| KRaft | No ZK dependency | Newer, less battle-tested | Future consideration |

**Source**: etcd, CockroachDB, TiKV all use Raft. Kafka moved from ZAB to KRaft.

### Storage Format

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Custom binary** | Optimal for logs | Implementation effort | **Selected** |
| RocksDB | Battle-tested LSM | Not optimized for logs | Rejected |
| SQLite | ACID, familiar | Overkill for append-only | Rejected |

**Source**: Kafka uses custom format. Pulsar uses BookKeeper (also custom).

### Replication Strategy

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **ISR (Kafka-style)** | Balances durability/availability | Complexity | **Selected** |
| Quorum writes | Simple | Either too slow or risky | Rejected |
| Async replication | Fast | Data loss on failure | Rejected |

**Source**: Kafka's ISR is industry standard for this trade-off.

---

## Further Reading

1. **The Log**: Kreps, J. "The Log: What every software engineer should know" (2013) - engineering.linkedin.com
2. **Raft Paper**: Ongaro, D. & Ousterhout, J. "In Search of an Understandable Consensus Algorithm" USENIX ATC (2014)
3. **Kafka Design**: kafka.apache.org/documentation/#design
4. **Kafka Internals**: Narkhede, N. et al. "Kafka: The Definitive Guide" O'Reilly (2017)
5. **Distributed Systems**: Kleppmann, M. "Designing Data-Intensive Applications" O'Reilly (2017) - Chapter 9
6. **KRaft**: Kafka Improvement Proposal KIP-500 "Replace ZooKeeper with a Self-Managed Metadata Quorum"
