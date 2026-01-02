"""Application coordinator for secure platform.

Orchestrates domain services and manages the security workflow:
- Certificate lifecycle (issuance, rotation, revocation)
- Policy management and authorization decisions
- Audit logging and compliance tracking

References:
    - design.md Section 2 (Architecture)
    - Hexagonal Architecture pattern
"""

from __future__ import annotations

import logging
from typing import Optional

from secure_platform.domain.entities.certificate import (
    Certificate,
    CertificateSigningRequest,
    CertificateState,
)
from secure_platform.domain.entities.audit import AuditEventType
from secure_platform.domain.entities.policy import (
    AuthorizationDecision,
    Effect,
    Policy,
    PolicyType,
    Principal,
    Resource,
)
from secure_platform.domain.services.audit_logger import AuditLogger
from secure_platform.domain.services.policy_engine import PolicyEngine
from secure_platform.domain.services.pki_manager import PKIManager
from secure_platform.domain.value_objects.identifiers import (
    create_spiffe_id,
)

logger = logging.getLogger(__name__)


class SecureCoordinator:
    """Orchestrates secure platform services.
    
    Manages the flow of certificate issuance, policy enforcement,
    and audit logging across the platform.
    """
    
    def __init__(self):
        """Initialize coordinator with services."""
        self._pki = PKIManager()
        self._policy_engine = PolicyEngine()
        self._audit_logger = AuditLogger()
        
        # Metrics (future: export to Prometheus)
        self._metrics = {
            "certs_issued": 0,
            "certs_rotated": 0,
            "certs_revoked": 0,
            "auth_allowed": 0,
            "auth_denied": 0,
            "policies_created": 0,
        }
    
    def initialize(self):
        """Initialize the secure platform."""
        logger.info("Initializing secure platform coordinator")
        # Add any startup logic here
    
    def request_certificate(
        self,
        requester: str,
        spiffe_id: str,
        public_key_pem: str,
        validity_days: int = 7,
    ) -> Certificate:
        """Request and issue a certificate.
        
        Args:
            requester: Who is requesting the cert.
            spiffe_id: SPIFFE ID for the certificate.
            public_key_pem: Public key in PEM format.
            validity_days: How long the cert is valid.
            
        Returns:
            Issued certificate.
        """
        logger.info(f"Certificate request from {requester} for {spiffe_id}")
        
        # Create CSR
        csr = CertificateSigningRequest(
            spiffe_id=spiffe_id,
            public_key_pem=public_key_pem,
            validity_days=validity_days,
        )
        
        # Issue certificate
        cert = self._pki.issue_certificate(csr)
        
        # Audit log
        self._audit_logger.log_cert_issued(
            requester,
            cert.specs.serial,
            spiffe_id,
        )
        
        # Record metric
        self._metrics["certs_issued"] += 1
        
        logger.info(f"Issued certificate {cert.specs.serial} for {spiffe_id}")
        return cert
    
    def rotate_certificate(
        self,
        requester: str,
        old_cert: Certificate,
    ) -> Certificate:
        """Rotate a certificate to a new one.
        
        Args:
            requester: Who is rotating the cert.
            old_cert: Current certificate to rotate out.
            
        Returns:
            New certificate.
        """
        logger.info(f"Certificate rotation request from {requester}")
        
        # Rotate to new cert
        new_cert = self._pki.rotate_certificate(old_cert)
        
        # Audit log
        self._audit_logger.log(
            event_type=AuditEventType.CERT_ROTATED,
            principal_id=requester,
            action="rotate",
            resource_id=old_cert.specs.spiffe_id,
            result="success",
            details={
                "old_serial": old_cert.specs.serial,
                "new_serial": new_cert.specs.serial,
            },
        )
        
        # Record metric
        self._metrics["certs_rotated"] += 1
        
        return new_cert
    
    def authorize_request(
        self,
        principal: Principal,
        action: str,
        resource: Resource,
    ) -> AuthorizationDecision:
        """Authorize an access request.

        Args:
            principal: Who is requesting access.
            action: What action they want to perform.
            resource: Resource being accessed (with attributes for ABAC).

        Returns:
            Authorization decision.
        """
        resource_id = str(resource.resource_id)
        logger.info(f"Authorization request: {principal.principal_id} {action} {resource_id}")

        # Evaluate against policies
        decision = self._policy_engine.evaluate(principal, action, resource)

        # Audit log
        if decision.allowed:
            self._audit_logger.log_auth_allowed(
                principal.principal_id,
                action,
                resource_id,
            )
            self._metrics["auth_allowed"] += 1
        else:
            self._audit_logger.log_auth_denied(
                principal.principal_id,
                action,
                resource_id,
                reason=decision.reason,
            )
            self._metrics["auth_denied"] += 1

        return decision
    
    def create_policy(
        self,
        creator: str,
        policy: Policy,
    ) -> Policy:
        """Create a new authorization policy.
        
        Args:
            creator: Who is creating the policy.
            policy: Policy to create.
            
        Returns:
            Created policy.
        """
        logger.info(f"Creating policy {policy.policy_name}")
        
        self._policy_engine.add_policy(policy)
        
        # Audit log
        self._audit_logger.log(
            event_type=AuditEventType.POLICY_CREATED,
            principal_id=creator,
            action="create",
            resource_id=str(policy.policy_id),
            result="success",
            details={"policy_type": policy.policy_type.value},
        )
        
        # Record metric
        self._metrics["policies_created"] += 1
        
        return policy
    
    def get_metrics(self) -> dict:
        """Get coordinator metrics.
        
        Returns:
            Dictionary of metrics.
        """
        return {
            **self._metrics,
            "audit_stats": self._audit_logger.get_stats(),
            "audit_root_hash": self._audit_logger.get_root_hash(),
        }
    
    def get_audit_stats(self) -> dict:
        """Get audit log statistics.
        
        Returns:
            Audit statistics.
        """
        return self._audit_logger.get_stats()
