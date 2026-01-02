# Security Agent Design Document

## High-Level Component Overview

This document describes a kernel-level security monitoring agent implementing the principles from Brendan Gregg's eBPF work, Falco's runtime security approach, and industry EDR (Endpoint Detection and Response) architectures.

### Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Security Agent                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  Detection   │───▶│   Response   │───▶│    Alert     │                   │
│  │   Engine     │    │   Engine     │    │   Manager    │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    Event Processing                               │       │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │       │
│  │  │   Rules    │  │  Baseline  │  │   ML       │  │  Dedup &   │  │       │
│  │  │   Engine   │  │  Learning  │  │  Anomaly   │  │  Throttle  │  │       │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    eBPF Probes (Kernel)                           │       │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │       │
│  │  │  Syscall   │  │  Network   │  │   File     │  │  Process   │  │       │
│  │  │  Tracing   │  │  Events    │  │   Events   │  │  Lifecycle │  │       │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Each Component Exists

| Component | Purpose | Why This Approach | Source |
|-----------|---------|-------------------|--------|
| **eBPF Probes** | Kernel visibility | Safe kernel instrumentation, no module required. <3% overhead [1] | Gregg "BPF Performance Tools" |
| **Syscall Tracing** | Behavior monitoring | All user-kernel interactions go through syscalls [2] | Linux kernel docs |
| **Baseline Learning** | Anomaly detection | Normal behavior varies per workload. ML adapts [3] | UEBA concept |
| **Rule Engine** | Known threat detection | Fast matching for known IOCs and TTPs [4] | Sigma rules, YARA |
| **Response Engine** | Automated remediation | Speed critical: malware spreads in seconds [5] | EDR architecture |
| **SIEM Integration** | Centralized logging | Correlation across hosts, compliance [6] | SOC best practices |

### Why eBPF?

```
Traditional monitoring:           eBPF approach:
┌─────────────────────┐          ┌─────────────────────┐
│  Kernel Module      │          │  eBPF Program       │
│  - Risky            │          │  - Safe (verifier)  │
│  - Version-specific │          │  - Portable (BTF)   │
│  - Can crash kernel │          │  - Can't crash      │
│  - Root required    │          │  - CAP_BPF only     │
└─────────────────────┘          └─────────────────────┘
         │                                │
         ▼                                ▼
  Kernel panics possible          Worst case: probe disabled
```

### References

1. **eBPF**: Gregg, B. "BPF Performance Tools" Addison-Wesley (2019)
2. **Syscall Tracing**: Linux Kernel Documentation "Syscall Auditing"
3. **UEBA**: Gartner "User and Entity Behavior Analytics" Market Guide
4. **Detection Rules**: Sigma - Generic Signature Format for SIEM Systems (github.com/SigmaHQ/sigma)
5. **EDR Architecture**: MITRE ATT&CK Framework - Endpoint Detection
6. **SIEM**: NIST SP 800-92 "Guide to Computer Security Log Management"

---

## 1. Problem Statement & Non-Goals

### Problem Statement

Build a kernel-level security monitoring agent using eBPF that provides real-time threat detection, behavioral analysis, and automated response capabilities.

**Why build this?** Understanding runtime security is essential for:
- Detecting fileless malware and living-off-the-land attacks
- Building defense-in-depth beyond perimeter security
- Compliance requirements (container runtime security)

### Goals

- Kernel-level visibility via eBPF probes (syscalls, network, file operations)
- ML-based anomaly detection with behavioral baselines
- Rule-based threat detection engine
- Automated response actions (quarantine, kill, isolate)
- SIEM integration for alert forwarding

### Non-Goals

- Full EDR replacement
- User-space antivirus scanning
- Cross-platform support (Linux only)
- GUI management console

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User Space                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │  Detection  │  │  Response   │  │    Alert                    │  │
│  │   Engine    │  │   Engine    │  │    Manager                  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                       Event Processing                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │   Event     │  │  Baseline   │  │    Rule                     │  │
│  │   Router    │  │   Manager   │  │    Engine                   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                          eBPF Layer                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │  Syscall    │  │  Network    │  │    File                     │  │
│  │  Probes     │  │  Probes     │  │    Probes                   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                         Kernel Space                                 │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Linux Kernel                                  ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### How Components Interact

1. **Event Flow**: Kernel event → eBPF probe → Ring buffer → User-space processing
2. **Detection Flow**: Event → Rules check → ML scoring → Alert generation
3. **Response Flow**: High-severity alert → Response engine → Kill/quarantine/isolate

---

## 3. Core Components & APIs

### eBPF Probe Manager

**How eBPF works**:
```
1. Write eBPF C program
2. Compile to BPF bytecode
3. Verifier checks safety (no loops, bounded memory)
4. JIT compiles to native code
5. Attach to kernel hook point
6. Events flow to user-space via ring buffer

┌─────────────────┐     ┌─────────────────┐
│  eBPF Program   │────▶│  BPF Verifier   │
│  (bytecode)     │     │  (safety check) │
└─────────────────┘     └────────┬────────┘
                                 │ safe
                                 ▼
┌─────────────────┐     ┌─────────────────┐
│   Ring Buffer   │◀────│  Kernel Hook    │
│  (to userspace) │     │  (kprobe/tp)    │
└─────────────────┘     └─────────────────┘
```

```python
class ProbeManager(Protocol):
    def load_probe(self, probe_type: ProbeType, config: ProbeConfig) -> ProbeHandle: ...
    def unload_probe(self, handle: ProbeHandle) -> None: ...
    def read_events(self, handle: ProbeHandle) -> AsyncIterator[Event]: ...
```

### Detection Engine

**Why both rules and ML?**
```
Rules (Sigma/YARA):             ML Anomaly Detection:
✓ Known threats (IOCs)          ✓ Unknown threats (0-day)
✓ Fast, deterministic           ✓ Adapts to environment
✓ Explainable                   ✓ Behavioral patterns
✗ Can't detect novel attacks    ✗ False positives
✗ Signature maintenance         ✗ Requires training

Combined approach: Rules first (fast reject), ML for unknowns
```

```python
class DetectionEngine(Protocol):
    def evaluate(self, event: Event) -> list[Detection]: ...
    def add_rule(self, rule: DetectionRule) -> None: ...
    def update_baseline(self, metrics: BaselineMetrics) -> None: ...
```

### Response Engine

**Why automated response?**
```
Manual response:                 Automated response:
  Detect → Alert → Human         Detect → Response → Alert

  Time: minutes to hours         Time: milliseconds

  Result: Malware spreads        Result: Contained immediately
```

**Response actions by severity**:
| Severity | Action | Example |
|----------|--------|---------|
| Critical | Kill + isolate | Ransomware detected |
| High | Kill process | Reverse shell |
| Medium | Quarantine file | Suspicious binary |
| Low | Alert only | Unusual network connection |

```python
class ResponseEngine(Protocol):
    def execute(self, detection: Detection) -> ResponseResult: ...
    def kill_process(self, pid: int) -> bool: ...
    def quarantine_file(self, path: Path) -> bool: ...
    def isolate_network(self, pid: int) -> bool: ...
```

---

## 4. Data Models & State Machines

### Detection State Machine

```
    ┌─────────────┐   threshold met   ┌─────────────┐
    │  LEARNING   │ ───────────────▶  │   ACTIVE    │
    └─────────────┘                   └─────────────┘
          │                                 │
          │ manual disable            alert triggered
          │                                 │
          ▼                                 ▼
    ┌─────────────┐                   ┌─────────────┐
    │  DISABLED   │                   │  ALERTING   │
    └─────────────┘                   └─────────────┘
                                            │
                                     response complete
                                            │
                                            ▼
                                      ┌─────────────┐
                                      │  RESOLVED   │───▶ Back to ACTIVE
                                      └─────────────┘
```

### Event Schema

**Why this schema?** Aligns with industry standards (ECS, OCSF) for SIEM compatibility.

```
Event:
├── timestamp: datetime (nanosecond precision)
├── event_type: enum {syscall, network, file, process}
├── process:
│   ├── pid: int
│   ├── ppid: int
│   ├── name: str
│   ├── exe_path: str (full path to binary)
│   ├── cmdline: str (arguments)
│   ├── uid: int
│   ├── gid: int
│   └── container_id: str | None
├── payload: union
│   ├── SyscallEvent: {syscall_nr, args, return_value}
│   ├── NetworkEvent: {src_ip, dst_ip, src_port, dst_port, protocol}
│   └── FileEvent: {path, operation, flags}
└── metadata:
    ├── hostname: str
    ├── kernel_version: str
    └── agent_version: str
```

---

## 5. Concurrency Model

| Component | Model | Rationale |
|-----------|-------|-----------|
| eBPF event reading | asyncio | High throughput (100K+ events/sec) |
| Detection engine | Thread pool | CPU-bound ML inference |
| Response actions | asyncio | I/O-bound system calls |
| Baseline computation | multiprocessing | CPU-intensive statistics |

### Event Processing Pipeline

```python
async def process_events(self):
    """
    High-throughput event pipeline.

    Target: 100,000 events/sec with <1ms latency
    """
    async for event in self.probe_manager.read_events():
        # Stage 1: Fast rule matching (O(1) hash lookup)
        if rule_match := self.rules.match(event):
            await self.handle_detection(rule_match)
            continue

        # Stage 2: Baseline check (fast statistical)
        if self.baseline.is_anomalous(event):
            # Stage 3: ML scoring (slower, only for anomalies)
            score = await self.ml_model.score(event)
            if score > self.threshold:
                await self.handle_detection(Detection(event, score))
```

---

## 6. Failure Modes & Recovery

| Failure | Detection | Recovery | Impact |
|---------|-----------|----------|--------|
| eBPF probe crash | Heartbeat timeout | Reload probe | Brief blind spot |
| Ring buffer overflow | Dropped event counter | Increase buffer, sample | Missed events |
| ML model failure | Exception handling | Fallback to rules only | Reduced detection |
| Response timeout | Watchdog timer | Log and escalate | Threat may persist |
| Agent crash | systemd watchdog | Auto-restart | Brief blind spot |

### Ring Buffer Sizing

```
Events per second: 100,000
Event size: ~500 bytes
Buffer duration target: 5 seconds

Required buffer: 100,000 × 500 × 5 = 250 MB

Actual: Use 512MB ring buffer
- Handles burst traffic (2x normal)
- Survives brief processing delays
```

---

## 7. Security Threat Model

### Agent Protection

| Threat | Mitigation | Implementation |
|--------|------------|----------------|
| Agent tampering | Process protection | Immutable binary, integrity checks |
| Evasion via rootkit | Multiple detection vectors | Combine syscall + network + file |
| DoS via event flood | Rate limiting, sampling | Drop events above threshold |
| Privilege escalation | Minimal capabilities | CAP_BPF, CAP_PERFMON only |
| Credential theft | No persistent creds | Use instance metadata |

### Detection Capabilities (MITRE ATT&CK)

| Tactic | Technique | Detection Method |
|--------|-----------|------------------|
| Execution | T1059 Command-Line | Syscall: execve with args |
| Persistence | T1053 Scheduled Task | File: /etc/cron.d writes |
| Privilege Escalation | T1068 Exploit | Syscall: uid change patterns |
| Defense Evasion | T1070 Log Deletion | File: /var/log modifications |
| Credential Access | T1003 Dumping | File: /etc/shadow access |
| Lateral Movement | T1021 Remote Services | Network: SSH/RDP connections |
| Exfiltration | T1048 Exfil Over Alt Protocol | Network: DNS tunneling patterns |

---

## 8. Performance Targets

| Metric | Target | How Achieved |
|--------|--------|--------------|
| Event processing latency | < 1ms p99 | Efficient ring buffer, async I/O |
| Detection latency | < 10ms p99 | Rules first, ML async |
| CPU overhead | < 2% | Kernel-efficient eBPF |
| Memory footprint | < 100MB | Streaming, no event storage |
| Events per second | > 100,000 | Parallel processing, batching |

### Overhead Breakdown

```
eBPF probe overhead:
- kprobe attachment: ~50ns per call
- Event copy to ring buffer: ~200ns
- Total per syscall: ~300ns

At 10,000 syscalls/sec: 3ms CPU time = 0.3% of one core

With 4 probes active: ~1.2% overhead typical
```

---

## 9. Operational Concerns

### Agent Management

```bash
# Start agent with config
security-agent start --config /etc/security_agent/config.yaml

# Check agent status and health
security-agent status

# Reload rules without restart
security-agent rules reload

# Reset baseline (after known-good state)
security-agent baseline reset
```

### Detection Rules

**Sigma-like rule format**:
```yaml
title: Suspicious Shell Execution from /tmp
id: a1b2c3d4-e5f6-7890-abcd-ef1234567890
status: production
description: Detects shell execution from temporary directory
severity: high
tags:
  - attack.execution
  - attack.t1059
detection:
  condition:
    event_type: syscall
    syscall: execve
    process.exe_path|startswith: /tmp/
    process.name|in: [bash, sh, zsh, dash]
response:
  action: kill_process
  alert: true
```

### Baseline Management

```bash
# View current baseline statistics
security-agent baseline show

# Export baseline for review
security-agent baseline export --output baseline.json

# Import baseline (e.g., from golden image)
security-agent baseline import --input baseline.json

# Learning mode (don't alert, just learn)
security-agent baseline learn --duration 24h
```

---

## 10. Alternatives Considered

### Kernel Instrumentation

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **eBPF** | Safe, fast, portable | Linux 4.9+ only | **Selected** |
| ptrace | Portable | High overhead (10x+) | Rejected |
| Audit subsystem | Stable, built-in | Limited visibility | Supplementary |
| Kernel module | Full access | Can crash kernel | Rejected |

**Source**: eBPF is used by Cilium, Falco, Datadog, and major cloud providers.

### Detection Approach

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Rules + ML** | Best of both | Complexity | **Selected** |
| Rules only | Simple, fast | Can't detect unknown | Insufficient |
| ML only | Adaptive | False positives, explainability | Insufficient |

**Source**: Modern EDRs (CrowdStrike, SentinelOne) all use hybrid approaches.

### Response Strategy

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Automated** | Fast response | Risk of false positive damage | **Selected** for high-severity |
| Alert only | Safe | Slow response | Default for low-severity |
| Human-in-loop | Controlled | Too slow for attacks | For medium-severity |

**Source**: NIST recommends automated response for critical threats.

---

## Further Reading

1. **eBPF**: Gregg, B. "BPF Performance Tools" Addison-Wesley (2019)
2. **Falco**: falco.org - Cloud-Native Runtime Security
3. **Sigma Rules**: github.com/SigmaHQ/sigma
4. **MITRE ATT&CK**: attack.mitre.org
5. **Linux Tracing**: brendangregg.com/linuxperf.html
6. **EDR Architecture**: Red Canary "Atomic Red Team" - Testing EDR
