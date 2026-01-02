"""Inbound ports - API contracts for the secure platform.

Inbound ports define the interfaces that clients and upper layers
use to interact with PKI, policy engine, and audit logging.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol

from secure_platform.domain.entities.certificate import (
    Certificate,
    CertificateSigningRequest,
)
from secure_platform.domain.entities.policy import (
    AuthorizationDecision,
    Policy,
    Principal,
    Resource,
)
from secure_platform.domain.entities.audit import (
    AuditEvent,
    AuditEventType,
    MerkleProof,
)
from secure_platform.domain.value_objects.identifiers import (
    CertificateSerial,
    SpiffeId,
)


# =============================================================================
# PKI Manager Port
# =============================================================================


@dataclass
class PKIStats:
    """Statistics for PKI manager monitoring."""

    total_issued: int
    valid_certs: int
    revoked_certs: int
    expired_certs: int
    unique_principals: int


class PKIManagerPort(Protocol):
    """Protocol for PKI (Public Key Infrastructure) operations.

    Handles X.509 certificate issuance, validation, rotation, and revocation
    for mTLS authentication.

    Thread Safety:
        All methods must be thread-safe.

    References:
        - RFC 5280 (X.509 PKI)
        - SPIFFE (spiffe.io)

    Example:
        csr = CertificateSigningRequest(spiffe_id, public_key_pem)
        cert = pki.issue_certificate(csr)
        if pki.validate_certificate(cert):
            # Use certificate
            pass
    """

    @abstractmethod
    def issue_certificate(self, csr: CertificateSigningRequest) -> Certificate:
        """Issue a new certificate from CSR.

        Args:
            csr: Certificate signing request.

        Returns:
            Issued certificate.

        Raises:
            PKIError: If CSR is invalid.
        """
        ...

    @abstractmethod
    def validate_certificate(self, cert: Certificate) -> bool:
        """Validate a certificate.

        Checks state, revocation, and expiry.

        Args:
            cert: Certificate to validate.

        Returns:
            True if certificate is valid.
        """
        ...

    @abstractmethod
    def revoke_certificate(self, serial: CertificateSerial, reason: str = "") -> None:
        """Revoke a certificate.

        Args:
            serial: Serial number of certificate to revoke.
            reason: Revocation reason.

        Raises:
            PKIError: If certificate not found.
        """
        ...

    @abstractmethod
    def rotate_certificate(self, old_cert: Certificate) -> Certificate:
        """Rotate a certificate to a new one.

        Creates a new certificate with the same SPIFFE ID.

        Args:
            old_cert: Certificate to rotate.

        Returns:
            New certificate.

        Raises:
            PKIError: If old certificate is invalid.
        """
        ...

    @abstractmethod
    def get_certificate(self, serial: CertificateSerial) -> Optional[Certificate]:
        """Get a certificate by serial number.

        Args:
            serial: Certificate serial number.

        Returns:
            Certificate or None if not found.
        """
        ...

    @abstractmethod
    def get_current_certificate(self, spiffe_id: SpiffeId) -> Optional[Certificate]:
        """Get current valid certificate for SPIFFE ID.

        Args:
            spiffe_id: SPIFFE ID of the principal.

        Returns:
            Current certificate or None.
        """
        ...

    @abstractmethod
    def get_stats(self) -> PKIStats:
        """Get PKI statistics for monitoring.

        Returns:
            PKI statistics.
        """
        ...


class PKIError(Exception):
    """Raised when PKI operation fails."""

    pass


# =============================================================================
# Policy Engine Port
# =============================================================================


@dataclass
class PolicyEngineStats:
    """Statistics for policy engine monitoring."""

    total_policies: int
    enabled_policies: int
    total_rules: int


class PolicyEnginePort(Protocol):
    """Protocol for authorization policy evaluation.

    Evaluates access control decisions using RBAC, ABAC, and ACL policies.

    Thread Safety:
        All methods must be thread-safe.

    Default Policy:
        Default DENY - explicit ALLOW required.

    References:
        - NIST SP 800-162 (ABAC Guide)

    Example:
        policy_engine.add_policy(admin_policy)
        decision = policy_engine.evaluate(principal, "read", resource)
        if decision.allowed:
            # Allow access
            pass
    """

    @abstractmethod
    def add_policy(self, policy: Policy) -> None:
        """Add an authorization policy.

        Args:
            policy: Policy to add.
        """
        ...

    @abstractmethod
    def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy.

        Args:
            policy_id: Policy ID to remove.

        Returns:
            True if policy was removed.
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        principal: Principal,
        action: str,
        resource: Resource,
    ) -> AuthorizationDecision:
        """Evaluate if principal can perform action on resource.

        Args:
            principal: Principal making the request.
            action: Action to perform (e.g., "read", "write").
            resource: Resource being accessed.

        Returns:
            AuthorizationDecision with allow/deny.
        """
        ...

    @abstractmethod
    def get_stats(self) -> PolicyEngineStats:
        """Get policy engine statistics.

        Returns:
            Policy engine statistics.
        """
        ...


class PolicyEngineError(Exception):
    """Raised when policy engine operation fails."""

    pass


# =============================================================================
# Audit Logger Port
# =============================================================================


@dataclass
class AuditLogStats:
    """Statistics for audit logger monitoring."""

    total_events: int
    events_by_type: dict[str, int]
    tree_root_hash: Optional[str]


class AuditLoggerPort(Protocol):
    """Protocol for append-only, tamper-evident audit logging.

    Provides Merkle tree-based audit logs for compliance and forensics.

    Thread Safety:
        All methods must be thread-safe.

    Immutability:
        Events cannot be modified or deleted once logged.

    References:
        - RFC 6962 (Certificate Transparency)

    Example:
        event = audit.log(event_type, principal, action, resource, result)
        # Later, verify integrity
        if audit.verify_event(event.event_id):
            # Event is intact
            pass
    """

    @abstractmethod
    def log(
        self,
        event_type: AuditEventType,
        principal_id: str,
        action: str,
        resource_id: str,
        result: str,
        details: Optional[dict] = None,
    ) -> AuditEvent:
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
        ...

    @abstractmethod
    def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """Get an audit event by ID.

        Args:
            event_id: Event ID.

        Returns:
            Event or None if not found.
        """
        ...

    @abstractmethod
    def get_proof(self, event_id: str) -> Optional[MerkleProof]:
        """Get Merkle proof for an event.

        Args:
            event_id: Event ID.

        Returns:
            Merkle proof or None.
        """
        ...

    @abstractmethod
    def verify_event(self, event_id: str) -> bool:
        """Verify an event hasn't been tampered with.

        Args:
            event_id: Event ID.

        Returns:
            True if event is verified.
        """
        ...

    @abstractmethod
    def get_root_hash(self) -> Optional[str]:
        """Get current Merkle tree root hash.

        Used to verify log integrity.

        Returns:
            Root hash or None.
        """
        ...

    @abstractmethod
    def get_stats(self) -> AuditLogStats:
        """Get audit log statistics.

        Returns:
            Audit log statistics.
        """
        ...


class AuditLogError(Exception):
    """Raised when audit log operation fails."""

    pass


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # PKI Manager
    "PKIManagerPort",
    "PKIStats",
    "PKIError",
    # Policy Engine
    "PolicyEnginePort",
    "PolicyEngineStats",
    "PolicyEngineError",
    # Audit Logger
    "AuditLoggerPort",
    "AuditLogStats",
    "AuditLogError",
]
