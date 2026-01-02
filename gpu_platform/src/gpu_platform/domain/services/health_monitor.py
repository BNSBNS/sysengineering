"""GPU health monitoring and alert management.

Monitors GPU health metrics against alert rules, generates alerts, and executes
remediation actions (throttle, preempt, quarantine).

References:
    - design.md Section 6 (Failure Modes & Recovery)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from gpu_platform.domain.entities.gpu_device import GPUDevice, GPUHealth, GPUState
from gpu_platform.domain.value_objects.alert_rules import (
    AlertRule,
    AlertSeverity,
    AlertType,
    DEFAULT_ALERT_RULES,
)
from gpu_platform.domain.value_objects.gpu_identifiers import GPUId

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Triggered alert with context."""
    alert_id: str
    rule_id: str
    alert_type: AlertType
    severity: AlertSeverity
    gpu_id: GPUId
    metric_name: str
    metric_value: float
    threshold: float
    description: str
    timestamp: float = field(default_factory=time.time)
    action_taken: Optional[str] = None


class HealthMonitor:
    """Monitor GPU health against alert rules and generate alerts."""
    
    def __init__(self, rules: Optional[list[AlertRule]] = None):
        """Initialize health monitor.
        
        Args:
            rules: Alert rules to use. Defaults to DEFAULT_ALERT_RULES.
        """
        self.rules = rules or DEFAULT_ALERT_RULES
        self._alerts: dict[str, Alert] = {}  # alert_id -> Alert
        self._metric_history: dict[GPUId, dict] = {}  # Track metric trends
        self._alert_callbacks: list[Callable[[Alert], None]] = []
        self._alert_counter = 0
    
    def register_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Register callback to be called when alerts are generated.
        
        Args:
            callback: Function that takes an Alert.
        """
        self._alert_callbacks.append(callback)
    
    def evaluate_gpu_health(self, device: GPUDevice) -> list[Alert]:
        """Evaluate GPU against all rules and generate alerts.
        
        Args:
            device: GPU device with current health metrics.
            
        Returns:
            List of new alerts triggered.
        """
        if device.health is None:
            return []
        
        new_alerts = []
        health = device.health
        
        # Track metric history for trend analysis
        if device.specs.gpu_id not in self._metric_history:
            self._metric_history[device.specs.gpu_id] = {}
        
        history = self._metric_history[device.specs.gpu_id]
        
        for rule in self.rules:
            metric_value = self._get_metric_value(health, rule.metric_name)
            
            if metric_value is None:
                continue
            
            # Check if rule threshold is violated
            if self._check_threshold(metric_value, rule.threshold, rule.comparison):
                # Track time metric has been in violation state
                violation_key = f"{rule.rule_id}_violation_time"
                
                if violation_key not in history:
                    history[violation_key] = time.time()
                
                violation_duration = time.time() - history[violation_key]
                
                # Generate alert if duration threshold met
                if rule.duration_seconds is None or violation_duration >= rule.duration_seconds:
                    alert = Alert(
                        alert_id=f"alert-{self._alert_counter}",
                        rule_id=rule.rule_id,
                        alert_type=rule.alert_type,
                        severity=rule.severity,
                        gpu_id=device.specs.gpu_id,
                        metric_name=rule.metric_name,
                        metric_value=metric_value,
                        threshold=rule.threshold,
                        description=rule.description,
                    )
                    self._alert_counter += 1
                    new_alerts.append(alert)
                    self._alerts[alert.alert_id] = alert
                    
                    # Notify callbacks
                    for callback in self._alert_callbacks:
                        callback(alert)
                    
                    logger.warning(
                        f"Alert triggered: {rule.alert_type.value} on {device.specs.gpu_id} "
                        f"({rule.metric_name}={metric_value} vs {rule.threshold})"
                    )
            else:
                # Clear violation timer when metric returns to normal
                violation_key = f"{rule.rule_id}_violation_time"
                if violation_key in history:
                    del history[violation_key]
        
        return new_alerts
    
    def get_active_alerts(self, gpu_id: Optional[GPUId] = None) -> list[Alert]:
        """Get active alerts.
        
        Args:
            gpu_id: Optional GPU ID to filter by.
            
        Returns:
            List of active alerts.
        """
        if gpu_id:
            return [a for a in self._alerts.values() if a.gpu_id == gpu_id]
        return list(self._alerts.values())
    
    def clear_alert(self, alert_id: str) -> bool:
        """Clear/acknowledge an alert.
        
        Args:
            alert_id: Alert to clear.
            
        Returns:
            True if alert was cleared.
        """
        if alert_id in self._alerts:
            del self._alerts[alert_id]
            return True
        return False
    
    def _get_metric_value(self, health: GPUHealth, metric_name: str) -> Optional[float]:
        """Extract metric value from health object.
        
        Args:
            health: GPU health metrics.
            metric_name: Name of metric (e.g., "temperature_c").
            
        Returns:
            Metric value or None if not found.
        """
        metric_map = {
            "temperature_c": health.temperature_c,
            "power_w": health.power_w,
            "utilization_percent": health.utilization_percent,
            "memory_used_mb": health.memory_used_mb,
            "memory_utilization_percent": (health.memory_used_mb / 40960 * 100),  # Estimate
            "ecc_errors_correctable": float(health.ecc_errors_correctable),
            "ecc_errors_uncorrectable": float(health.ecc_errors_uncorrectable),
            "throttled": 1.0 if health.throttled else 0.0,
        }
        return metric_map.get(metric_name)
    
    def _check_threshold(
        self, value: float, threshold: float, comparison: str
    ) -> bool:
        """Check if metric value violates threshold.
        
        Args:
            value: Metric value.
            threshold: Threshold to compare against.
            comparison: "gt", "gte", "lt", "lte", "eq".
            
        Returns:
            True if threshold is violated.
        """
        if comparison == "gt":
            return value > threshold
        elif comparison == "gte":
            return value >= threshold
        elif comparison == "lt":
            return value < threshold
        elif comparison == "lte":
            return value <= threshold
        elif comparison == "eq":
            return value == threshold
        return False


class AlertManager:
    """Manage alerts and execute remediation actions."""
    
    def __init__(self, monitor: HealthMonitor):
        """Initialize alert manager.
        
        Args:
            monitor: Health monitor to track alerts from.
        """
        self.monitor = monitor
        self._remediation_actions: dict[AlertType, Callable] = {}
        self._alert_history: list[Alert] = []
        
        # Register default remediation actions
        monitor.register_alert_callback(self._on_alert_generated)
    
    def register_remediation_action(
        self, alert_type: AlertType, action: Callable[[Alert, GPUDevice], None]
    ) -> None:
        """Register remediation action for alert type.
        
        Args:
            alert_type: Type of alert to handle.
            action: Function to execute when alert is triggered.
        """
        self._remediation_actions[alert_type] = action
    
    def _on_alert_generated(self, alert: Alert) -> None:
        """Called when alert is generated.
        
        Args:
            alert: The alert that was generated.
        """
        self._alert_history.append(alert)
        logger.info(f"Alert recorded: {alert.alert_type.value} on {alert.gpu_id}")
    
    def execute_remediation(self, alert: Alert, device: GPUDevice) -> bool:
        """Execute remediation action for alert.
        
        Args:
            alert: Alert to remediate.
            device: GPU device with issue.
            
        Returns:
            True if remediation was executed.
        """
        action_func = self._remediation_actions.get(alert.alert_type)
        
        if not action_func:
            logger.warning(f"No remediation registered for {alert.alert_type.value}")
            return False
        
        try:
            action_func(alert, device)
            alert.action_taken = alert.alert_type.value
            logger.info(f"Remediation executed for {alert.alert_id}: {alert.action_taken}")
            return True
        except Exception as e:
            logger.error(f"Failed to execute remediation: {e}")
            return False
    
    def get_alert_history(self, gpu_id: Optional[GPUId] = None, limit: int = 100) -> list[Alert]:
        """Get historical alerts.
        
        Args:
            gpu_id: Optional GPU ID to filter by.
            limit: Maximum number of alerts to return.
            
        Returns:
            List of alerts, most recent first.
        """
        alerts = self._alert_history
        
        if gpu_id:
            alerts = [a for a in alerts if a.gpu_id == gpu_id]
        
        return list(reversed(alerts))[-limit:]
