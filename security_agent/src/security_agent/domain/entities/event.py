"""Security event entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class EventType(str, Enum):
    """Security event types."""
    SYSCALL = "syscall"
    NETWORK = "network"
    FILE = "file"
    PROCESS = "process"


class ProcessInfo:
    """Process information."""
    
    def __init__(
        self,
        pid: int,
        ppid: int,
        uid: int,
        gid: int,
        comm: str,
        exe: str = "",
    ):
        self.pid = pid
        self.ppid = ppid
        self.uid = uid
        self.gid = gid
        self.comm = comm
        self.exe = exe


class NetworkConnection:
    """Network connection information."""
    
    def __init__(
        self,
        protocol: str,
        src_ip: str,
        src_port: int,
        dst_ip: str,
        dst_port: int,
    ):
        self.protocol = protocol
        self.src_ip = src_ip
        self.src_port = src_port
        self.dst_ip = dst_ip
        self.dst_port = dst_port


@dataclass
class SecurityEvent:
    """Security event from eBPF probes."""
    
    event_id: str
    event_type: EventType
    timestamp: datetime
    process: ProcessInfo
    syscall: Optional[str] = None
    syscall_args: Optional[dict] = None
    network: Optional[NetworkConnection] = None
    file_path: Optional[str] = None
    severity: str = "info"  # info, warning, critical
    raw_data: dict = field(default_factory=dict)
    
    def is_critical(self) -> bool:
        """Check if event is critical severity."""
        return self.severity == "critical"
    
    def get_context(self) -> dict:
        """Get event context for analysis."""
        return {
            "pid": self.process.pid,
            "uid": self.process.uid,
            "syscall": self.syscall,
            "file": self.file_path,
        }
