"""Secure Platform domain layer."""

from secure_platform.domain.entities.audit import AuditEvent, AuditEventType, MerkleTree
from secure_platform.domain.entities.certificate import (
    Certificate,
    CertificateSigningRequest,
    CertificateState,
)
from secure_platform.domain.entities.policy import (
    AuthorizationDecision,
    Effect,
    Policy,
    PolicyType,
    Principal,
    Resource,
    Role,
)
from secure_platform.domain.services.audit_logger import AuditLogger
from secure_platform.domain.services.policy_engine import PolicyEngine
from secure_platform.domain.services.pki_manager import PKIManager
from secure_platform.domain.value_objects.identifiers import (
    CertificateSerial,
    PolicyId,
    PrincipalId,
    ResourceId,
    SpiffeId,
    create_policy_id,
    create_principal_id,
    create_resource_id,
    create_spiffe_id,
)

__all__ = [
    # Value objects
    "SpiffeId",
    "CertificateSerial",
    "PolicyId",
    "PrincipalId",
    "ResourceId",
    "create_spiffe_id",
    "create_policy_id",
    "create_principal_id",
    "create_resource_id",
    # Entities
    "AuditEvent",
    "AuditEventType",
    "MerkleTree",
    "Certificate",
    "CertificateSigningRequest",
    "CertificateState",
    "AuthorizationDecision",
    "Effect",
    "Policy",
    "PolicyType",
    "Principal",
    "Resource",
    "Role",
    # Services
    "AuditLogger",
    "PolicyEngine",
    "PKIManager",
]
