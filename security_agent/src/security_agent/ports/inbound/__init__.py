"""Inbound ports - API contracts for the security agent.

Inbound ports define the interfaces that clients and upper layers
use to interact with threat detection and automated response.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol

from security_agent.domain.entities.event import SecurityEvent
from security_agent.domain.entities.rule import DetectionRule


# =============================================================================
# Detection Engine Port
# =============================================================================


@dataclass
class DetectionResult:
    """Result of threat detection."""

    event_id: str
    rule_id: str
    rule_name: str
    threat_score: float
    severity: str
    timestamp: float
    process_info: dict


@dataclass
class DetectionEngineStats:
    """Statistics for detection engine monitoring."""

    total_rules: int
    events_processed: int
    detections_found: int
    baseline_keys: int


class DetectionEnginePort(Protocol):
    """Protocol for threat detection operations.

    Analyzes security events against detection rules and baseline statistics.

    Thread Safety:
        All methods must be thread-safe.

    Performance:
        Target: <3% CPU overhead.

    Example:
        engine.add_rule(suspicious_execve_rule)
        result = engine.detect(security_event)
        if result:
            # Handle detection
            pass
    """

    @abstractmethod
    def add_rule(self, rule: DetectionRule) -> None:
        """Add a detection rule.

        Args:
            rule: Rule to add.
        """
        ...

    @abstractmethod
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a detection rule.

        Args:
            rule_id: ID of rule to remove.

        Returns:
            True if rule was removed.
        """
        ...

    @abstractmethod
    def detect(self, event: SecurityEvent) -> Optional[DetectionResult]:
        """Detect threats in a security event.

        Args:
            event: Security event to analyze.

        Returns:
            Detection result if threat found, None otherwise.
        """
        ...

    @abstractmethod
    def update_baseline(self, key: str, value: float) -> None:
        """Update baseline statistics.

        Args:
            key: Baseline key (e.g., "syscall_rate").
            value: Value to add to baseline.
        """
        ...

    @abstractmethod
    def is_anomaly(self, key: str, value: float, threshold: float = 2.0) -> bool:
        """Check if value is anomalous based on baseline.

        Uses z-score based anomaly detection.

        Args:
            key: Baseline key.
            value: Value to check.
            threshold: Standard deviations for anomaly.

        Returns:
            True if anomaly detected.
        """
        ...

    @abstractmethod
    def get_stats(self) -> DetectionEngineStats:
        """Get detection engine statistics.

        Returns:
            Detection engine statistics.
        """
        ...


# =============================================================================
# Response Engine Port
# =============================================================================


@dataclass
class ResponseAction:
    """Action to take in response to a threat."""

    action_type: str  # "alert", "log", "isolate", "kill_process", "quarantine"
    target: str  # Process ID, file path, etc.
    severity: str
    reason: str


@dataclass
class ResponseEngineStats:
    """Statistics for response engine monitoring."""

    total_policies: int
    actions_executed: int
    actions_by_type: dict[str, int]


class ResponseEnginePort(Protocol):
    """Protocol for automated threat response.

    Executes response actions based on detection results.

    Thread Safety:
        All methods must be thread-safe.

    Cooldown:
        Actions have cooldown periods to prevent response storms.

    Example:
        response_engine.add_policy("high_severity", kill_process_action)
        result = response_engine.execute_response(detection_result)
    """

    @abstractmethod
    def add_policy(self, severity: str, action: ResponseAction) -> None:
        """Add a response policy.

        Args:
            severity: Severity level this policy applies to.
            action: Action to take.
        """
        ...

    @abstractmethod
    def remove_policy(self, severity: str) -> bool:
        """Remove a response policy.

        Args:
            severity: Severity level to remove policy for.

        Returns:
            True if policy was removed.
        """
        ...

    @abstractmethod
    def execute_response(self, detection: DetectionResult) -> Optional[ResponseAction]:
        """Execute response for a detection.

        Args:
            detection: Detection result to respond to.

        Returns:
            Action executed, or None if no action taken.
        """
        ...

    @abstractmethod
    def is_on_cooldown(self, target: str) -> bool:
        """Check if target is on cooldown.

        Args:
            target: Target identifier (e.g., process ID).

        Returns:
            True if target is on cooldown.
        """
        ...

    @abstractmethod
    def get_stats(self) -> ResponseEngineStats:
        """Get response engine statistics.

        Returns:
            Response engine statistics.
        """
        ...


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Detection Engine
    "DetectionEnginePort",
    "DetectionResult",
    "DetectionEngineStats",
    # Response Engine
    "ResponseEnginePort",
    "ResponseAction",
    "ResponseEngineStats",
]
