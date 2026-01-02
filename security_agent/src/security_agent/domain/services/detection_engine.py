"""Detection engine service."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from security_agent.domain.entities.event import SecurityEvent
from security_agent.domain.entities.rule import DetectionRule


class DetectionEngine:
    """Core threat detection logic."""
    
    def __init__(self, rules: list[DetectionRule] | None = None):
        """Initialize detection engine.
        
        Args:
            rules: List of detection rules.
        """
        self.rules = rules or []
        self.baseline_stats = {}
    
    def add_rule(self, rule: DetectionRule) -> None:
        """Add detection rule.
        
        Args:
            rule: Rule to add.
        """
        self.rules.append(rule)
    
    def detect(self, event: SecurityEvent) -> dict | None:
        """Detect threats in event.
        
        Args:
            event: Security event to analyze.
            
        Returns:
            Detection result if threat found, None otherwise.
        """
        max_score = 0.0
        matched_rule = None
        
        event_dict = {
            "pid": event.process.pid,
            "uid": event.process.uid,
            "syscall": event.syscall,
            "file": event.file_path,
            "type": event.event_type.value,
        }
        
        # Check all rules
        for rule in self.rules:
            score = rule.get_score(event_dict)
            if score > max_score:
                max_score = score
                matched_rule = rule
        
        # If matched rule above threshold, return detection
        if matched_rule and max_score >= 0.8:
            return {
                "event_id": event.event_id,
                "rule_id": matched_rule.rule_id,
                "rule_name": matched_rule.name,
                "threat_score": max_score,
                "severity": matched_rule.severity.value,
                "timestamp": event.timestamp,
                "process_info": {
                    "pid": event.process.pid,
                    "comm": event.process.comm,
                    "uid": event.process.uid,
                },
            }
        
        return None
    
    def update_baseline(self, key: str, value: float) -> None:
        """Update baseline statistics.
        
        Args:
            key: Baseline key.
            value: Baseline value.
        """
        if key not in self.baseline_stats:
            self.baseline_stats[key] = []
        
        self.baseline_stats[key].append(value)
        
        # Keep only last 1000 entries
        if len(self.baseline_stats[key]) > 1000:
            self.baseline_stats[key] = self.baseline_stats[key][-1000:]
    
    def is_anomaly(self, key: str, value: float, threshold: float = 2.0) -> bool:
        """Check if value is anomalous based on baseline.
        
        Args:
            key: Baseline key.
            value: Value to check.
            threshold: Standard deviations for anomaly.
            
        Returns:
            True if anomaly detected.
        """
        if key not in self.baseline_stats or len(self.baseline_stats[key]) < 10:
            return False
        
        values = self.baseline_stats[key]
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        stddev = variance ** 0.5
        
        if stddev == 0:
            return False
        
        z_score = abs((value - mean) / stddev)
        return z_score > threshold
