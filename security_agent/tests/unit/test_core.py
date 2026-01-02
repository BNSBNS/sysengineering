"""Unit tests for Security Agent detection and response engines."""

import pytest
import time
from datetime import datetime
from unittest.mock import patch

from security_agent.domain.entities.event import (
    SecurityEvent,
    EventType,
    ProcessInfo,
)
from security_agent.domain.entities.rule import (
    DetectionRule,
    RuleType,
    RuleSeverity,
)
from security_agent.domain.services.detection_engine import DetectionEngine
from security_agent.domain.services.response_engine import (
    ResponseEngine,
    ResponsePolicy,
    ResponseAction,
)


@pytest.mark.unit
class TestDetectionEngine:
    """Test detection engine functionality."""

    def create_test_event(
        self,
        event_id: str = "evt-001",
        syscall: str = "execve",
        file_path: str = "/etc/passwd",
        pid: int = 1234,
        uid: int = 0,
    ) -> SecurityEvent:
        """Create test security event."""
        process = ProcessInfo(
            pid=pid,
            ppid=1,
            uid=uid,
            gid=0,
            comm="test_process",
            exe="/usr/bin/test",
        )
        return SecurityEvent(
            event_id=event_id,
            event_type=EventType.SYSCALL,
            timestamp=datetime.utcnow(),
            process=process,
            syscall=syscall,
            file_path=file_path,
        )

    def test_add_rule(self):
        """Test adding detection rules."""
        engine = DetectionEngine()
        assert len(engine.rules) == 0

        rule = DetectionRule(
            rule_id="rule-001",
            name="Test Rule",
            description="Test description",
            rule_type=RuleType.SIGNATURE,
            severity=RuleSeverity.WARNING,
        )
        engine.add_rule(rule)

        assert len(engine.rules) == 1
        assert engine.rules[0].rule_id == "rule-001"

    def test_detect_matching_rule(self):
        """Test detection with matching rule."""
        engine = DetectionEngine()
        rule = DetectionRule(
            rule_id="rule-001",
            name="Passwd Access",
            description="Detects /etc/passwd access",
            rule_type=RuleType.SIGNATURE,
            severity=RuleSeverity.CRITICAL,
            pattern="/etc/passwd",
            threshold=0.9,
        )
        engine.add_rule(rule)

        event = self.create_test_event(file_path="/etc/passwd")
        result = engine.detect(event)

        assert result is not None
        assert result["rule_id"] == "rule-001"
        assert result["severity"] == "critical"
        assert result["threat_score"] >= 0.8

    def test_detect_no_matching_rule(self):
        """Test detection with no matching rule."""
        engine = DetectionEngine()
        rule = DetectionRule(
            rule_id="rule-001",
            name="Shadow Access",
            description="Detects /etc/shadow access",
            rule_type=RuleType.SIGNATURE,
            severity=RuleSeverity.CRITICAL,
            pattern="/etc/shadow",
        )
        engine.add_rule(rule)

        event = self.create_test_event(file_path="/var/log/syslog")
        result = engine.detect(event)

        assert result is None

    def test_detect_disabled_rule(self):
        """Test detection skips disabled rules."""
        engine = DetectionEngine()
        rule = DetectionRule(
            rule_id="rule-001",
            name="Passwd Access",
            description="Detects /etc/passwd access",
            rule_type=RuleType.SIGNATURE,
            severity=RuleSeverity.CRITICAL,
            pattern="/etc/passwd",
            enabled=False,
        )
        engine.add_rule(rule)

        event = self.create_test_event(file_path="/etc/passwd")
        result = engine.detect(event)

        assert result is None

    def test_detect_highest_score_rule(self):
        """Test detection returns highest scoring rule."""
        engine = DetectionEngine()
        rule1 = DetectionRule(
            rule_id="rule-001",
            name="Low Score Rule",
            description="Low score",
            rule_type=RuleType.SIGNATURE,
            severity=RuleSeverity.WARNING,
            pattern="passwd",
            threshold=0.8,
        )
        rule2 = DetectionRule(
            rule_id="rule-002",
            name="High Score Rule",
            description="High score",
            rule_type=RuleType.SIGNATURE,
            severity=RuleSeverity.CRITICAL,
            pattern="/etc/passwd",
            threshold=0.95,
        )
        engine.add_rule(rule1)
        engine.add_rule(rule2)

        event = self.create_test_event(file_path="/etc/passwd")
        result = engine.detect(event)

        assert result is not None
        assert result["rule_id"] == "rule-002"
        assert result["threat_score"] == 0.95

    def test_update_baseline(self):
        """Test baseline statistics updates."""
        engine = DetectionEngine()

        engine.update_baseline("cpu_usage", 50.0)
        engine.update_baseline("cpu_usage", 55.0)
        engine.update_baseline("cpu_usage", 60.0)

        assert "cpu_usage" in engine.baseline_stats
        assert len(engine.baseline_stats["cpu_usage"]) == 3
        assert engine.baseline_stats["cpu_usage"] == [50.0, 55.0, 60.0]

    def test_update_baseline_limit(self):
        """Test baseline keeps only last 1000 entries."""
        engine = DetectionEngine()

        for i in range(1100):
            engine.update_baseline("metric", float(i))

        assert len(engine.baseline_stats["metric"]) == 1000
        assert engine.baseline_stats["metric"][0] == 100.0  # First 100 were removed

    def test_is_anomaly_insufficient_data(self):
        """Test anomaly detection with insufficient baseline data."""
        engine = DetectionEngine()

        for i in range(5):
            engine.update_baseline("metric", float(i))

        assert not engine.is_anomaly("metric", 100.0)

    def test_is_anomaly_no_baseline(self):
        """Test anomaly detection with no baseline."""
        engine = DetectionEngine()
        assert not engine.is_anomaly("unknown_metric", 100.0)

    def test_is_anomaly_detected(self):
        """Test anomaly detection when value is anomalous."""
        engine = DetectionEngine()

        # Add baseline values around 50
        for i in range(100):
            engine.update_baseline("metric", 50.0 + (i % 5))

        # Check extreme value
        assert engine.is_anomaly("metric", 200.0, threshold=2.0)

    def test_is_anomaly_normal(self):
        """Test anomaly detection when value is normal."""
        engine = DetectionEngine()

        # Add baseline values around 50
        for i in range(100):
            engine.update_baseline("metric", 50.0 + (i % 5))

        # Check normal value
        assert not engine.is_anomaly("metric", 52.0, threshold=2.0)


@pytest.mark.unit
class TestResponseEngine:
    """Test response engine functionality."""

    def test_add_policy(self):
        """Test adding response policies."""
        engine = ResponseEngine()
        policy = ResponsePolicy(
            severity="critical",
            action=ResponseAction.KILL_PROCESS,
        )
        engine.add_policy("critical", policy)

        assert "critical" in engine.policies
        assert engine.policies["critical"].action == ResponseAction.KILL_PROCESS

    def test_execute_response_no_policy(self):
        """Test execute response with no matching policy."""
        engine = ResponseEngine()
        detection = {
            "event_id": "evt-001",
            "severity": "critical",
        }
        result = engine.execute_response(detection)
        assert result is None

    def test_execute_response_disabled_policy(self):
        """Test execute response with disabled policy."""
        engine = ResponseEngine()
        policy = ResponsePolicy(
            severity="critical",
            action=ResponseAction.KILL_PROCESS,
            enabled=False,
        )
        engine.add_policy("critical", policy)

        detection = {
            "event_id": "evt-001",
            "severity": "critical",
        }
        result = engine.execute_response(detection)
        assert result is None

    def test_execute_response_alert(self):
        """Test execute alert response."""
        engine = ResponseEngine()
        policy = ResponsePolicy(
            severity="warning",
            action=ResponseAction.ALERT,
        )
        engine.add_policy("warning", policy)

        detection = {
            "event_id": "evt-001",
            "severity": "warning",
        }
        result = engine.execute_response(detection)

        assert result is not None
        assert result["status"] == "alerted"

    def test_execute_response_log(self):
        """Test execute log response."""
        engine = ResponseEngine()
        policy = ResponsePolicy(
            severity="info",
            action=ResponseAction.LOG,
        )
        engine.add_policy("info", policy)

        detection = {
            "event_id": "evt-002",
            "severity": "info",
        }
        result = engine.execute_response(detection)

        assert result is not None
        assert result["status"] == "logged"

    def test_execute_response_kill_process(self):
        """Test execute kill process response."""
        engine = ResponseEngine()
        policy = ResponsePolicy(
            severity="critical",
            action=ResponseAction.KILL_PROCESS,
        )
        engine.add_policy("critical", policy)

        detection = {
            "event_id": "evt-003",
            "severity": "critical",
            "process_info": {"pid": 1234},
        }
        result = engine.execute_response(detection)

        assert result is not None
        assert result["status"] == "killed"
        assert "1234" in result["message"]

    def test_execute_response_quarantine(self):
        """Test execute quarantine response."""
        engine = ResponseEngine()
        policy = ResponsePolicy(
            severity="critical",
            action=ResponseAction.QUARANTINE,
        )
        engine.add_policy("critical", policy)

        detection = {
            "event_id": "evt-004",
            "severity": "critical",
            "file_path": "/tmp/malware.exe",
        }
        result = engine.execute_response(detection)

        assert result is not None
        assert result["status"] == "quarantined"
        assert "/tmp/malware.exe" in result["message"]

    def test_response_log_recorded(self):
        """Test that responses are logged."""
        engine = ResponseEngine()
        policy = ResponsePolicy(
            severity="warning",
            action=ResponseAction.ALERT,
        )
        engine.add_policy("warning", policy)

        detection = {
            "event_id": "evt-005",
            "severity": "warning",
        }
        engine.execute_response(detection)

        assert len(engine.response_log) == 1
        assert engine.response_log[0]["event_id"] == "evt-005"
        assert engine.response_log[0]["action"] == "alert"

    def test_cooldown_blocks_duplicate_response(self):
        """Test cooldown prevents duplicate responses."""
        engine = ResponseEngine()
        policy = ResponsePolicy(
            severity="warning",
            action=ResponseAction.ALERT,
            cooldown_seconds=60,
        )
        engine.add_policy("warning", policy)

        detection = {
            "event_id": "evt-006",
            "severity": "warning",
        }

        # First response should succeed
        result1 = engine.execute_response(detection)
        assert result1 is not None

        # Second response should be blocked by cooldown
        result2 = engine.execute_response(detection)
        assert result2 is None

    def test_cooldown_expires(self):
        """Test cooldown expiration allows new response."""
        engine = ResponseEngine()
        policy = ResponsePolicy(
            severity="warning",
            action=ResponseAction.ALERT,
            cooldown_seconds=1,  # 1 second cooldown
        )
        engine.add_policy("warning", policy)

        detection = {
            "event_id": "evt-007",
            "severity": "warning",
        }

        # First response
        result1 = engine.execute_response(detection)
        assert result1 is not None

        # Wait for cooldown to expire
        time.sleep(1.1)

        # Second response should succeed after cooldown
        result2 = engine.execute_response(detection)
        assert result2 is not None

    def test_different_events_no_cooldown_interference(self):
        """Test cooldown doesn't affect different events."""
        engine = ResponseEngine()
        policy = ResponsePolicy(
            severity="warning",
            action=ResponseAction.ALERT,
            cooldown_seconds=60,
        )
        engine.add_policy("warning", policy)

        detection1 = {
            "event_id": "evt-008",
            "severity": "warning",
        }
        detection2 = {
            "event_id": "evt-009",
            "severity": "warning",
        }

        # Both events should get responses
        result1 = engine.execute_response(detection1)
        result2 = engine.execute_response(detection2)

        assert result1 is not None
        assert result2 is not None
        assert len(engine.response_log) == 2

    def test_isolate_action(self):
        """Test isolate action response."""
        engine = ResponseEngine()
        policy = ResponsePolicy(
            severity="critical",
            action=ResponseAction.ISOLATE,
        )
        engine.add_policy("critical", policy)

        detection = {
            "event_id": "evt-010",
            "severity": "critical",
            "process_info": {"pid": 5678},
        }
        result = engine.execute_response(detection)

        assert result is not None
        assert result["status"] == "isolated"
        assert "5678" in result["message"]
