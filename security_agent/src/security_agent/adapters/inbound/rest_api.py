"""FastAPI REST adapter for the Security Agent.

Provides HTTP endpoints for threat detection and automated response.

Usage:
    from security_agent.adapters.inbound.rest_api import create_app

    app = create_app()
    # Run with: uvicorn module:app --host 0.0.0.0 --port 8080

References:
    - design.md Section 2.4 (API Services)
    - ports/inbound/api.py (API contracts)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException, status
    from pydantic import BaseModel, Field

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

from security_agent.domain.entities.event import (
    EventType,
    ProcessInfo,
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


# Pydantic models for request/response serialization
if _HAS_FASTAPI:

    class ProcessInfoModel(BaseModel):
        """Process information."""

        pid: int = Field(..., description="Process ID")
        ppid: int = Field(default=1, description="Parent process ID")
        uid: int = Field(default=0, description="User ID")
        gid: int = Field(default=0, description="Group ID")
        comm: str = Field(..., description="Process command name")
        exe: str = Field(default="", description="Executable path")

    class SecurityEventRequest(BaseModel):
        """Security event to analyze."""

        event_id: str = Field(..., description="Unique event identifier")
        event_type: str = Field(..., description="Event type: syscall, network, file, process")
        process: ProcessInfoModel
        syscall: Optional[str] = Field(default=None, description="Syscall name if applicable")
        file_path: Optional[str] = Field(default=None, description="File path if applicable")
        severity: str = Field(default="info", description="Event severity")

    class DetectionResponse(BaseModel):
        """Detection result."""

        detected: bool
        event_id: str
        rule_id: Optional[str] = None
        rule_name: Optional[str] = None
        threat_score: Optional[float] = None
        severity: Optional[str] = None
        response_executed: bool = False
        response_result: Optional[dict] = None

    class AddRuleRequest(BaseModel):
        """Request to add a detection rule."""

        rule_id: str = Field(..., description="Unique rule identifier")
        name: str = Field(..., description="Rule name")
        description: str = Field(default="", description="Rule description")
        rule_type: str = Field(default="signature", description="Type: signature, anomaly, behavioral")
        severity: str = Field(default="warning", description="Severity: info, warning, critical")
        pattern: Optional[str] = Field(default=None, description="Pattern to match")
        threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Detection threshold")
        enabled: bool = Field(default=True, description="Whether rule is enabled")

    class AddPolicyRequest(BaseModel):
        """Request to add a response policy."""

        severity: str = Field(..., description="Severity: info, warning, critical")
        action: str = Field(..., description="Action: alert, log, isolate, kill_process, quarantine")
        enabled: bool = Field(default=True, description="Whether policy is enabled")
        cooldown_seconds: int = Field(default=60, ge=1, description="Cooldown between responses")

    class BaselineUpdateRequest(BaseModel):
        """Request to update baseline statistics."""

        key: str = Field(..., description="Baseline metric key")
        value: float = Field(..., description="Metric value")

    class AnomalyCheckRequest(BaseModel):
        """Request to check for anomaly."""

        key: str = Field(..., description="Baseline metric key")
        value: float = Field(..., description="Value to check")
        threshold: float = Field(default=2.0, ge=0.1, description="Standard deviation threshold")

    class AnomalyCheckResponse(BaseModel):
        """Anomaly check result."""

        key: str
        value: float
        is_anomaly: bool
        threshold: float

    class RuleListResponse(BaseModel):
        """List of detection rules."""

        rules: list[dict]
        count: int

    class PolicyListResponse(BaseModel):
        """List of response policies."""

        policies: list[dict]
        count: int

    class ResponseLogResponse(BaseModel):
        """Response log entries."""

        entries: list[dict]
        count: int

    class MetricsResponse(BaseModel):
        """Agent metrics."""

        rules_count: int
        policies_count: int
        response_log_count: int
        baseline_keys: list[str]

    class HealthResponse(BaseModel):
        """Health check response."""

        status: str
        version: str = "0.1.0"


def create_app(
    detection_engine: DetectionEngine | None = None,
    response_engine: ResponseEngine | None = None,
) -> "FastAPI":
    """Create FastAPI application with Security Agent endpoints.

    Args:
        detection_engine: Optional DetectionEngine instance (creates new if None).
        response_engine: Optional ResponseEngine instance (creates new if None).

    Returns:
        Configured FastAPI application.

    Raises:
        ImportError: If FastAPI is not installed.
    """
    if not _HAS_FASTAPI:
        raise ImportError(
            "FastAPI not installed. Install with: pip install fastapi uvicorn"
        )

    # Initialize engines
    detection = detection_engine or DetectionEngine()
    response = response_engine or ResponseEngine()

    # Type mappings
    RULE_TYPE_MAP = {
        "signature": RuleType.SIGNATURE,
        "anomaly": RuleType.ANOMALY,
        "behavioral": RuleType.BEHAVIORAL,
    }
    SEVERITY_MAP = {
        "info": RuleSeverity.INFO,
        "warning": RuleSeverity.WARNING,
        "critical": RuleSeverity.CRITICAL,
    }
    EVENT_TYPE_MAP = {
        "syscall": EventType.SYSCALL,
        "network": EventType.NETWORK,
        "file": EventType.FILE,
        "process": EventType.PROCESS,
    }
    ACTION_MAP = {
        "alert": ResponseAction.ALERT,
        "log": ResponseAction.LOG,
        "isolate": ResponseAction.ISOLATE,
        "kill_process": ResponseAction.KILL_PROCESS,
        "quarantine": ResponseAction.QUARANTINE,
    }

    app = FastAPI(
        title="Security Agent API",
        description="eBPF-based runtime security monitoring with threat detection and automated response",
        version="1.0.0",
    )

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Check agent health status."""
        return HealthResponse(status="healthy")

    @app.get("/metrics", response_model=MetricsResponse, tags=["System"])
    async def get_metrics():
        """Get agent metrics."""
        return MetricsResponse(
            rules_count=len(detection.rules),
            policies_count=len(response.policies),
            response_log_count=len(response.response_log),
            baseline_keys=list(detection.baseline_stats.keys()),
        )

    @app.post(
        "/detect",
        response_model=DetectionResponse,
        tags=["Detection"],
    )
    async def detect_threat(request: SecurityEventRequest):
        """Analyze a security event for threats and optionally respond."""
        # Parse event type
        event_type = EVENT_TYPE_MAP.get(request.event_type.lower())
        if not event_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid event_type: {request.event_type}. Must be: syscall, network, file, process",
            )

        # Build ProcessInfo
        process = ProcessInfo(
            pid=request.process.pid,
            ppid=request.process.ppid,
            uid=request.process.uid,
            gid=request.process.gid,
            comm=request.process.comm,
            exe=request.process.exe,
        )

        # Build SecurityEvent
        event = SecurityEvent(
            event_id=request.event_id,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            process=process,
            syscall=request.syscall,
            file_path=request.file_path,
            severity=request.severity,
        )

        # Run detection
        detection_result = detection.detect(event)

        if detection_result is None:
            return DetectionResponse(
                detected=False,
                event_id=request.event_id,
            )

        # Execute response if detection found
        response_result = response.execute_response(detection_result)

        return DetectionResponse(
            detected=True,
            event_id=request.event_id,
            rule_id=detection_result.get("rule_id"),
            rule_name=detection_result.get("rule_name"),
            threat_score=detection_result.get("threat_score"),
            severity=detection_result.get("severity"),
            response_executed=response_result is not None,
            response_result=response_result,
        )

    @app.post(
        "/rules",
        response_model=dict,
        status_code=status.HTTP_201_CREATED,
        tags=["Rules"],
    )
    async def add_rule(request: AddRuleRequest):
        """Add a detection rule."""
        rule_type = RULE_TYPE_MAP.get(request.rule_type.lower())
        if not rule_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid rule_type: {request.rule_type}. Must be: signature, anomaly, behavioral",
            )

        severity = SEVERITY_MAP.get(request.severity.lower())
        if not severity:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid severity: {request.severity}. Must be: info, warning, critical",
            )

        rule = DetectionRule(
            rule_id=request.rule_id,
            name=request.name,
            description=request.description,
            rule_type=rule_type,
            severity=severity,
            pattern=request.pattern,
            threshold=request.threshold,
            enabled=request.enabled,
        )
        detection.add_rule(rule)

        return {"rule_id": request.rule_id, "status": "created"}

    @app.get("/rules", response_model=RuleListResponse, tags=["Rules"])
    async def list_rules():
        """List all detection rules."""
        rules = [
            {
                "rule_id": r.rule_id,
                "name": r.name,
                "description": r.description,
                "rule_type": r.rule_type.value,
                "severity": r.severity.value,
                "pattern": r.pattern,
                "threshold": r.threshold,
                "enabled": r.enabled,
            }
            for r in detection.rules
        ]
        return RuleListResponse(rules=rules, count=len(rules))

    @app.post(
        "/policies",
        response_model=dict,
        status_code=status.HTTP_201_CREATED,
        tags=["Policies"],
    )
    async def add_policy(request: AddPolicyRequest):
        """Add a response policy."""
        action = ACTION_MAP.get(request.action.lower())
        if not action:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid action: {request.action}. Must be: alert, log, isolate, kill_process, quarantine",
            )

        policy = ResponsePolicy(
            severity=request.severity,
            action=action,
            enabled=request.enabled,
            cooldown_seconds=request.cooldown_seconds,
        )
        response.add_policy(request.severity, policy)

        return {"severity": request.severity, "action": request.action, "status": "created"}

    @app.get("/policies", response_model=PolicyListResponse, tags=["Policies"])
    async def list_policies():
        """List all response policies."""
        policies = [
            {
                "severity": severity,
                "action": p.action.value,
                "enabled": p.enabled,
                "cooldown_seconds": p.cooldown_seconds,
            }
            for severity, p in response.policies.items()
        ]
        return PolicyListResponse(policies=policies, count=len(policies))

    @app.get("/response-log", response_model=ResponseLogResponse, tags=["Response"])
    async def get_response_log():
        """Get response action log."""
        return ResponseLogResponse(
            entries=response.response_log,
            count=len(response.response_log),
        )

    @app.post("/baseline", response_model=dict, tags=["Baseline"])
    async def update_baseline(request: BaselineUpdateRequest):
        """Update baseline statistics for anomaly detection."""
        detection.update_baseline(request.key, request.value)
        baseline_count = len(detection.baseline_stats.get(request.key, []))
        return {
            "key": request.key,
            "value": request.value,
            "total_samples": baseline_count,
        }

    @app.post("/baseline/check-anomaly", response_model=AnomalyCheckResponse, tags=["Baseline"])
    async def check_anomaly(request: AnomalyCheckRequest):
        """Check if a value is anomalous based on baseline."""
        is_anomaly = detection.is_anomaly(request.key, request.value, request.threshold)
        return AnomalyCheckResponse(
            key=request.key,
            value=request.value,
            is_anomaly=is_anomaly,
            threshold=request.threshold,
        )

    return app
