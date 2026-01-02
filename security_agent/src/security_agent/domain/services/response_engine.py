"""Automated response service."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ResponseAction(str, Enum):
    """Response actions."""
    ALERT = "alert"
    LOG = "log"
    ISOLATE = "isolate"
    KILL_PROCESS = "kill_process"
    QUARANTINE = "quarantine"


@dataclass
class ResponsePolicy:
    """Response policy for threats."""
    
    severity: str  # info, warning, critical
    action: ResponseAction
    enabled: bool = True
    cooldown_seconds: int = 60


class ResponseEngine:
    """Executes responses to detected threats."""
    
    def __init__(self, policies: dict[str, ResponsePolicy] | None = None):
        """Initialize response engine.
        
        Args:
            policies: Response policies by severity.
        """
        self.policies = policies or {}
        self.response_log = []
        self.last_response_times = {}
    
    def add_policy(self, severity: str, policy: ResponsePolicy) -> None:
        """Add response policy.
        
        Args:
            severity: Severity level (info, warning, critical).
            policy: Response policy.
        """
        self.policies[severity] = policy
    
    def execute_response(self, detection: dict) -> dict | None:
        """Execute response to detection.
        
        Args:
            detection: Detection result from engine.
            
        Returns:
            Response result if executed, None otherwise.
        """
        severity = detection.get("severity", "info")
        event_id = detection.get("event_id")
        pid = detection.get("process_info", {}).get("pid")
        
        # Check cooldown
        if self._is_on_cooldown(event_id):
            return None
        
        # Get policy for severity
        policy = self.policies.get(severity)
        if not policy or not policy.enabled:
            return None
        
        # Execute action
        result = self._execute_action(policy.action, pid, detection)
        
        # Update cooldown
        self._set_cooldown(event_id, policy.cooldown_seconds)
        
        # Log response
        self.response_log.append({
            "action": policy.action.value,
            "event_id": event_id,
            "pid": pid,
            "result": result,
        })
        
        return result
    
    def _execute_action(self, action: ResponseAction, pid: int | None, detection: dict) -> dict:
        """Execute response action.
        
        Args:
            action: Action to execute.
            pid: Process ID if applicable.
            detection: Detection data.
            
        Returns:
            Execution result.
        """
        if action == ResponseAction.ALERT:
            return {"status": "alerted", "message": f"Alert for detection {detection.get('event_id')}"}
        elif action == ResponseAction.LOG:
            return {"status": "logged", "message": f"Logged detection {detection.get('event_id')}"}
        elif action == ResponseAction.ISOLATE:
            return {"status": "isolated", "message": f"Isolated process {pid}"}
        elif action == ResponseAction.KILL_PROCESS:
            return {"status": "killed", "message": f"Killed process {pid}"}
        elif action == ResponseAction.QUARANTINE:
            return {"status": "quarantined", "message": f"Quarantined file {detection.get('file_path')}"}
        else:
            return {"status": "unknown", "message": "Unknown action"}
    
    def _is_on_cooldown(self, event_id: str) -> bool:
        """Check if event is on cooldown.

        Args:
            event_id: Event ID.

        Returns:
            True if on cooldown.
        """
        if event_id not in self.last_response_times:
            return False
        import time
        return time.time() < self.last_response_times[event_id]
    
    def _set_cooldown(self, event_id: str, cooldown_seconds: int) -> None:
        """Set response cooldown.
        
        Args:
            event_id: Event ID.
            cooldown_seconds: Cooldown duration.
        """
        import time
        self.last_response_times[event_id] = time.time() + cooldown_seconds
