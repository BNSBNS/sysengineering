"""Audit logging service with Merkle tree integrity.

Provides append-only, tamper-evident audit logs for compliance and forensics.

References:
    - design.md Section 3 (Audit Logger)
    - RFC 6962 (Certificate Transparency)
"""

from __future__ import annotations

import logging
from typing import Optional

from secure_platform.domain.entities.audit import (
    AuditEvent,
    AuditEventType,
    MerkleProof,
    MerkleTree,
)

logger = logging.getLogger(__name__)


class AuditLogError(Exception):
    """Audit log operation failed."""
    pass


class AuditLogger:
    """Append-only audit log with Merkle tree."""
    
    def __init__(self):
        """Initialize audit logger."""
        self._tree = MerkleTree()
        self._events: dict[str, AuditEvent] = {}
        self._event_counter = 0
    
    def log(self, event_type: AuditEventType, principal_id: str, action: str, resource_id: str, result: str, details: Optional[dict] = None) -> AuditEvent:
        """Log an audit event.
        
        Args:
            event_type: Type of event.
            principal_id: Who performed the action.
            action: What they did.
            resource_id: What it affected.
            result: "success" or "failure".
            details: Additional details.
            
        Returns:
            Logged event.
        """
        event_id = f"event-{self._event_counter}"
        self._event_counter += 1
        
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            principal_id=principal_id,
            action=action,
            resource_id=resource_id,
            result=result,
            details=details or {},
        )
        
        # Append to Merkle tree
        self._tree.append(event)
        self._events[event_id] = event
        
        logger.info(
            f"Logged {event_type.value}: {principal_id} {action} {resource_id} ({result})"
        )
        
        return event
    
    def log_cert_issued(self, principal_id: str, cert_serial: str, spiffe_id: str) -> AuditEvent:
        """Log certificate issuance.
        
        Args:
            principal_id: Who requested the cert.
            cert_serial: Certificate serial number.
            spiffe_id: SPIFFE ID of the cert.
            
        Returns:
            Logged event.
        """
        return self.log(
            event_type=AuditEventType.CERT_ISSUED,
            principal_id=principal_id,
            action="issue",
            resource_id=spiffe_id,
            result="success",
            details={'cert_serial': cert_serial},
        )
    
    def log_auth_allowed(self, principal_id: str, action: str, resource_id: str) -> AuditEvent:
        """Log allowed authorization.
        
        Args:
            principal_id: Principal making request.
            action: Action allowed.
            resource_id: Resource accessed.
            
        Returns:
            Logged event.
        """
        return self.log(
            event_type=AuditEventType.AUTH_ALLOWED,
            principal_id=principal_id,
            action=action,
            resource_id=resource_id,
            result="success",
        )
    
    def log_auth_denied(self, principal_id: str, action: str, resource_id: str, reason: str = "") -> AuditEvent:
        """Log denied authorization.
        
        Args:
            principal_id: Principal making request.
            action: Action denied.
            resource_id: Resource accessed.
            reason: Denial reason.
            
        Returns:
            Logged event.
        """
        return self.log(
            event_type=AuditEventType.AUTH_DENIED,
            principal_id=principal_id,
            action=action,
            resource_id=resource_id,
            result="failure",
            details={'reason': reason},
        )
    
    def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """Get an audit event by ID.
        
        Args:
            event_id: Event ID.
            
        Returns:
            Event or None if not found.
        """
        return self._events.get(event_id)
    
    def get_proof(self, event_id: str) -> Optional[MerkleProof]:
        """Get Merkle proof for an event.
        
        Args:
            event_id: Event ID.
            
        Returns:
            Merkle proof or None.
        """
        return self._tree.get_proof(event_id)
    
    def verify_event(self, event_id: str) -> bool:
        """Verify an event hasn't been tampered with.
        
        Args:
            event_id: Event ID.
            
        Returns:
            True if event is verified.
        """
        event = self._events.get(event_id)
        if not event:
            return False
        
        proof = self._tree.get_proof(event_id)
        if not proof:
            return False
        
        root_hash = self._tree.get_root_hash()
        if not root_hash:
            return False
        
        return self._tree.verify_proof(proof, root_hash)
    
    def get_root_hash(self) -> Optional[str]:
        """Get current Merkle tree root hash.
        
        Used to verify log integrity.
        
        Returns:
            Root hash or None.
        """
        return self._tree.get_root_hash()
    
    def get_stats(self) -> dict:
        """Get audit log statistics.
        
        Returns:
            Dictionary with stats.
        """
        event_types = {}
        for event in self._events.values():
            event_type = event.event_type.value
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        return {
            'total_events': len(self._events),
            'events_by_type': event_types,
            'tree_root_hash': self.get_root_hash(),
        }
