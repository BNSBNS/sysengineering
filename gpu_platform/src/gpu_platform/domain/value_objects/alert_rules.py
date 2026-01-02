"""Alert rules for GPU health monitoring.

Defines thresholds and conditions that trigger alerts when GPU health metrics
exceed safe operating parameters. Rules are evaluated by the AlertManager service.

References:
    - design.md Section 6 (Failure Modes & Recovery)
    - NVIDIA GPU Best Practices Guide
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """Alert types for different failure modes."""
    THERMAL_WARNING = "thermal_warning"      # Temp > 80°C
    THERMAL_CRITICAL = "thermal_critical"    # Temp > 85°C
    POWER_SPIKE = "power_spike"               # Power > 400W
    POWER_CRITICAL = "power_critical"         # Power > 450W
    ECC_ERRORS_SPIKE = "ecc_errors_spike"     # Correctable ECC increased
    ECC_UNCORRECTABLE = "ecc_uncorrectable"   # Uncorrectable ECC errors
    THROTTLING = "throttling"                 # GPU thermal throttling
    MEMORY_PRESSURE = "memory_pressure"       # Memory utilization > 90%
    STALLED_JOB = "stalled_job"              # Job running too long without progress
    UNKNOWN_ERROR = "unknown_error"           # Unclassified error


@dataclass
class AlertRule:
    """Single alert rule with threshold and action."""
    rule_id: str
    alert_type: AlertType
    severity: AlertSeverity
    metric_name: str
    threshold: float
    comparison: str  # "gt" (>), "gte" (>=), "lt" (<), "lte" (<=), "eq" (==)
    duration_seconds: Optional[int] = None  # How long condition must persist before alerting
    action: str = "notify"  # "notify", "throttle", "preempt", "quarantine"
    description: str = ""


# Default alert rules for production monitoring
DEFAULT_ALERT_RULES = [
    # Temperature alerts
    AlertRule(
        rule_id="temp_warning",
        alert_type=AlertType.THERMAL_WARNING,
        severity=AlertSeverity.WARNING,
        metric_name="temperature_c",
        threshold=80.0,
        comparison="gte",
        duration_seconds=60,
        action="notify",
        description="GPU temperature warning (>=80°C for 60s)",
    ),
    AlertRule(
        rule_id="temp_critical",
        alert_type=AlertType.THERMAL_CRITICAL,
        severity=AlertSeverity.CRITICAL,
        metric_name="temperature_c",
        threshold=85.0,
        comparison="gt",
        duration_seconds=30,
        action="throttle",
        description="GPU critical temperature (>85°C for 30s) - throttle workload",
    ),
    
    # Power alerts
    AlertRule(
        rule_id="power_spike",
        alert_type=AlertType.POWER_SPIKE,
        severity=AlertSeverity.WARNING,
        metric_name="power_w",
        threshold=400.0,
        comparison="gt",
        duration_seconds=45,
        action="notify",
        description="Power consumption spike (>400W for 45s)",
    ),
    AlertRule(
        rule_id="power_critical",
        alert_type=AlertType.POWER_CRITICAL,
        severity=AlertSeverity.CRITICAL,
        metric_name="power_w",
        threshold=450.0,
        comparison="gte",
        duration_seconds=20,
        action="preempt",
        description="Critical power draw (>=450W for 20s) - preempt running job",
    ),
    
    # ECC error alerts
    AlertRule(
        rule_id="ecc_spike",
        alert_type=AlertType.ECC_ERRORS_SPIKE,
        severity=AlertSeverity.WARNING,
        metric_name="ecc_errors_correctable",
        threshold=10.0,  # 10+ correctable errors
        comparison="gte",
        duration_seconds=300,
        action="notify",
        description="ECC errors trending up (>=10 correctable in 5m window)",
    ),
    AlertRule(
        rule_id="ecc_uncorrectable",
        alert_type=AlertType.ECC_UNCORRECTABLE,
        severity=AlertSeverity.CRITICAL,
        metric_name="ecc_errors_uncorrectable",
        threshold=1.0,
        comparison="gte",
        duration_seconds=0,
        action="quarantine",
        description="Uncorrectable ECC error detected - quarantine GPU immediately",
    ),
    
    # Throttling alert
    AlertRule(
        rule_id="throttling",
        alert_type=AlertType.THROTTLING,
        severity=AlertSeverity.WARNING,
        metric_name="throttled",
        threshold=1.0,
        comparison="eq",
        duration_seconds=30,
        action="notify",
        description="GPU thermal throttling active (>30s) - reduce job pressure",
    ),
    
    # Memory pressure
    AlertRule(
        rule_id="memory_pressure",
        alert_type=AlertType.MEMORY_PRESSURE,
        severity=AlertSeverity.WARNING,
        metric_name="memory_utilization_percent",
        threshold=90.0,
        comparison="gte",
        duration_seconds=60,
        action="notify",
        description="High memory utilization (>=90% for 60s)",
    ),
]
