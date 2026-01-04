"""Integration tests for security agent detection and response workflow."""

import pytest
from datetime import datetime

from security_agent.domain.entities.event import (
    EventType,
    ProcessInfo,
    NetworkConnection,
    SecurityEvent,
)
from security_agent.domain.entities.rule import (
    DetectionRule,
    RuleSeverity,
    RuleType,
)
from security_agent.domain.services.detection_engine import DetectionEngine
from security_agent.domain.services.response_engine import (
    ResponseAction,
    ResponseEngine,
    ResponsePolicy,
)


def create_test_event(
    event_id: str = "evt-001",
    event_type: EventType = EventType.SYSCALL,
    pid: int = 1234,
    uid: int = 0,
    syscall: str = "execve",
    file_path: str = "/tmp/malicious.sh",
) -> SecurityEvent:
    """Create a test security event."""
    process = ProcessInfo(
        pid=pid,
        ppid=1,
        uid=uid,
        gid=uid,
        comm="suspicious",
        exe="/bin/sh",
    )
    return SecurityEvent(
        event_id=event_id,
        event_type=event_type,
        timestamp=datetime.now(),
        process=process,
        syscall=syscall,
        file_path=file_path,
    )


class TestDetectionEngineIntegration:
    """Integration tests for detection engine."""

    def test_rule_matching(self):
        """Test rule matching against events."""
        engine = DetectionEngine()

        # Add detection rule
        rule = DetectionRule(
            rule_id="rule-001",
            name="Suspicious Execve",
            description="Detect suspicious execve syscalls",
            rule_type=RuleType.SIGNATURE,
            severity=RuleSeverity.CRITICAL,
            pattern="execve",
            threshold=0.9,
        )
        engine.add_rule(rule)

        # Create matching event
        event = create_test_event(syscall="execve")
        result = engine.detect(event)

        assert result is not None
        assert result["rule_id"] == "rule-001"
        assert result["severity"] == "critical"
        assert result["threat_score"] >= 0.8

    def test_no_match_returns_none(self):
        """Test that non-matching events return None."""
        engine = DetectionEngine()

        rule = DetectionRule(
            rule_id="rule-001",
            name="Specific Pattern",
            description="Match specific pattern",
            rule_type=RuleType.SIGNATURE,
            severity=RuleSeverity.WARNING,
            pattern="specific_malware_signature",
            threshold=0.8,
        )
        engine.add_rule(rule)

        # Create non-matching event
        event = create_test_event(syscall="read", file_path="/etc/passwd")
        result = engine.detect(event)

        assert result is None

    def test_multiple_rules_highest_score(self):
        """Test that highest scoring rule is returned."""
        engine = DetectionEngine()

        # Add low severity rule
        rule1 = DetectionRule(
            rule_id="rule-low",
            name="Low Severity",
            description="Low severity match",
            rule_type=RuleType.SIGNATURE,
            severity=RuleSeverity.INFO,
            pattern="execve",
            threshold=0.5,
        )
        engine.add_rule(rule1)

        # Add high severity rule
        rule2 = DetectionRule(
            rule_id="rule-high",
            name="High Severity",
            description="High severity match",
            rule_type=RuleType.SIGNATURE,
            severity=RuleSeverity.CRITICAL,
            pattern="execve",
            threshold=0.95,
        )
        engine.add_rule(rule2)

        event = create_test_event(syscall="execve")
        result = engine.detect(event)

        assert result is not None
        assert result["rule_id"] == "rule-high"
        assert result["threat_score"] == 0.95


class TestAnomalyDetection:
    """Integration tests for anomaly detection."""

    def test_baseline_learning(self):
        """Test that baseline is learned from normal activity."""
        engine = DetectionEngine()

        # Simulate normal syscall rates
        for i in range(100):
            engine.update_baseline("syscall_rate", 50 + (i % 10))

        # Check that baseline exists
        assert "syscall_rate" in engine.baseline_stats
        assert len(engine.baseline_stats["syscall_rate"]) == 100

    def test_anomaly_detection(self):
        """Test anomaly detection based on baseline."""
        engine = DetectionEngine()

        # Build baseline with normal values around 100
        for _ in range(100):
            engine.update_baseline("syscall_rate", 100)

        # Normal value should not be anomaly
        assert not engine.is_anomaly("syscall_rate", 100)
        assert not engine.is_anomaly("syscall_rate", 105)

        # Extremely high value should be anomaly
        assert engine.is_anomaly("syscall_rate", 1000, threshold=2.0)

    def test_insufficient_data_no_anomaly(self):
        """Test that insufficient baseline data doesn't flag anomalies."""
        engine = DetectionEngine()

        # Only 5 data points (less than minimum 10)
        for i in range(5):
            engine.update_baseline("new_metric", i)

        # Should not detect anomaly with insufficient data
        assert not engine.is_anomaly("new_metric", 1000)


class TestResponseEngineIntegration:
    """Integration tests for response engine."""

    def test_policy_execution(self):
        """Test response policy execution."""
        engine = ResponseEngine()

        # Add policies
        engine.add_policy("critical", ResponsePolicy(
            severity="critical",
            action=ResponseAction.KILL_PROCESS,
            cooldown_seconds=0,
        ))
        engine.add_policy("warning", ResponsePolicy(
            severity="warning",
            action=ResponseAction.ALERT,
            cooldown_seconds=0,
        ))

        # Critical detection
        detection = {
            "event_id": "evt-001",
            "severity": "critical",
            "process_info": {"pid": 1234},
        }
        result = engine.execute_response(detection)

        assert result is not None
        assert result["status"] == "killed"

    def test_cooldown_prevents_repeated_response(self):
        """Test that cooldown prevents duplicate responses."""
        engine = ResponseEngine()

        engine.add_policy("critical", ResponsePolicy(
            severity="critical",
            action=ResponseAction.ALERT,
            cooldown_seconds=60,  # 60 second cooldown
        ))

        detection = {
            "event_id": "evt-001",
            "severity": "critical",
            "process_info": {"pid": 1234},
        }

        # First response executes
        result1 = engine.execute_response(detection)
        assert result1 is not None

        # Second response blocked by cooldown
        result2 = engine.execute_response(detection)
        assert result2 is None

    def test_no_policy_no_response(self):
        """Test that missing policy results in no response."""
        engine = ResponseEngine()

        # No policies configured
        detection = {
            "event_id": "evt-001",
            "severity": "info",
            "process_info": {"pid": 1234},
        }

        result = engine.execute_response(detection)
        assert result is None


class TestEndToEndDetectionResponse:
    """Integration tests for complete detection-to-response workflow."""

    def test_complete_workflow(self):
        """Test complete detection and response workflow."""
        # Setup
        detection_engine = DetectionEngine()
        response_engine = ResponseEngine()

        # Add detection rule
        detection_engine.add_rule(DetectionRule(
            rule_id="root-shell",
            name="Root Shell Execution",
            description="Detect root shell execution",
            rule_type=RuleType.BEHAVIORAL,
            severity=RuleSeverity.CRITICAL,
            pattern="execve",
            threshold=0.9,
        ))

        # Add response policy
        response_engine.add_policy("critical", ResponsePolicy(
            severity="critical",
            action=ResponseAction.ISOLATE,
            cooldown_seconds=0,
        ))

        # Simulate suspicious event
        event = create_test_event(
            event_id="evt-suspicious-001",
            syscall="execve",
            uid=0,  # Root user
            file_path="/bin/sh",
        )

        # Detect
        detection = detection_engine.detect(event)
        assert detection is not None
        assert detection["severity"] == "critical"

        # Respond
        response = response_engine.execute_response(detection)
        assert response is not None
        assert response["status"] == "isolated"

    def test_multi_event_detection(self):
        """Test processing multiple events."""
        detection_engine = DetectionEngine()
        response_engine = ResponseEngine()

        # Setup rules
        detection_engine.add_rule(DetectionRule(
            rule_id="file-access",
            name="Sensitive File Access",
            description="Detect access to sensitive files",
            rule_type=RuleType.SIGNATURE,
            severity=RuleSeverity.WARNING,
            pattern="/etc/shadow",
            threshold=0.85,
        ))

        response_engine.add_policy("warning", ResponsePolicy(
            severity="warning",
            action=ResponseAction.LOG,
            cooldown_seconds=0,
        ))

        # Process multiple events
        events = [
            create_test_event(event_id=f"evt-{i}", file_path=f"/tmp/file{i}")
            for i in range(5)
        ]
        # Add one suspicious event
        events.append(create_test_event(
            event_id="evt-suspicious",
            file_path="/etc/shadow",
        ))

        detections = []
        for event in events:
            result = detection_engine.detect(event)
            if result:
                detections.append(result)

        assert len(detections) == 1
        assert detections[0]["event_id"] == "evt-suspicious"


class TestResponseLogging:
    """Integration tests for response logging."""

    def test_response_log_accumulation(self):
        """Test that responses are logged."""
        engine = ResponseEngine()

        engine.add_policy("warning", ResponsePolicy(
            severity="warning",
            action=ResponseAction.ALERT,
            cooldown_seconds=0,
        ))

        # Execute multiple responses
        for i in range(5):
            detection = {
                "event_id": f"evt-{i}",
                "severity": "warning",
                "process_info": {"pid": 1000 + i},
            }
            engine.execute_response(detection)

        assert len(engine.response_log) == 5

        # Check log contents
        for i, log_entry in enumerate(engine.response_log):
            assert log_entry["event_id"] == f"evt-{i}"
            assert log_entry["action"] == "alert"


class TestRuleSeverityLevels:
    """Integration tests for severity-based detection."""

    def test_severity_based_routing(self):
        """Test that detections route to correct severity policies."""
        detection_engine = DetectionEngine()
        response_engine = ResponseEngine()

        # Add rules with different severities
        detection_engine.add_rule(DetectionRule(
            rule_id="info-rule",
            name="Info Level",
            description="Info detection",
            rule_type=RuleType.SIGNATURE,
            severity=RuleSeverity.INFO,
            pattern="info_pattern",
            threshold=0.8,
        ))

        detection_engine.add_rule(DetectionRule(
            rule_id="critical-rule",
            name="Critical Level",
            description="Critical detection",
            rule_type=RuleType.SIGNATURE,
            severity=RuleSeverity.CRITICAL,
            pattern="critical_pattern",
            threshold=0.95,
        ))

        # Add response policies
        response_engine.add_policy("info", ResponsePolicy(
            severity="info",
            action=ResponseAction.LOG,
            cooldown_seconds=0,
        ))
        response_engine.add_policy("critical", ResponsePolicy(
            severity="critical",
            action=ResponseAction.KILL_PROCESS,
            cooldown_seconds=0,
        ))

        # Info event
        info_event = create_test_event(
            event_id="info-evt",
            file_path="info_pattern_here",
        )
        info_detection = detection_engine.detect(info_event)
        if info_detection:
            info_response = response_engine.execute_response(info_detection)
            assert info_response["status"] == "logged"

        # Critical event
        critical_event = create_test_event(
            event_id="critical-evt",
            file_path="critical_pattern_here",
        )
        critical_detection = detection_engine.detect(critical_event)
        if critical_detection:
            critical_response = response_engine.execute_response(critical_detection)
            assert critical_response["status"] == "killed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
