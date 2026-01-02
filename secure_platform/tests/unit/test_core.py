"""Secure Platform unit tests."""

import pytest
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
    PolicyId,
    PolicyRule,
    PolicyType,
    Principal,
    PrincipalId,
    Resource,
    ResourceId,
    Role,
)
from secure_platform.domain.services.audit_logger import AuditLogger
from secure_platform.domain.services.policy_engine import PolicyEngine
from secure_platform.domain.services.pki_manager import PKIManager
from secure_platform.domain.value_objects.identifiers import (
    create_policy_id,
    create_principal_id,
    create_resource_id,
    create_spiffe_id,
)


class TestIdentifiers:
    """Test identity identifier creation."""
    
    def test_create_spiffe_id(self):
        """Test SPIFFE ID creation."""
        spiffe_id = create_spiffe_id("service", "payment")
        assert "spiffe://" in spiffe_id
        assert "payment" in spiffe_id
    
    def test_create_policy_id(self):
        """Test policy ID creation."""
        policy_id = create_policy_id("reader_policy")
        assert "policy-" in policy_id


class TestCertificates:
    """Test certificate entity and lifecycle."""
    
    def test_certificate_creation(self):
        """Test creating a certificate from CSR."""
        csr = CertificateSigningRequest(
            spiffe_id=create_spiffe_id("service", "api"),
            public_key_pem="-----BEGIN PUBLIC KEY-----\nMIIBIjANBg...",  # Mock key
            validity_days=7,
        )
        
        cert = csr.create_certificate()
        assert cert.state == CertificateState.VALID
        assert cert.is_valid
    
    def test_certificate_revocation(self):
        """Test certificate revocation."""
        csr = CertificateSigningRequest(
            spiffe_id=create_spiffe_id("service", "db"),
            public_key_pem="-----BEGIN PUBLIC KEY-----\nMIIBIjANBg...",
            validity_days=7,
        )
        
        cert = csr.create_certificate()
        cert.revoke("key compromised")
        
        assert cert.state == CertificateState.REVOKED
        assert not cert.is_valid


class TestPKIManager:
    """Test PKI certificate management."""
    
    def test_issue_certificate(self):
        """Test issuing a certificate."""
        pki = PKIManager()
        
        csr = CertificateSigningRequest(
            spiffe_id=create_spiffe_id("service", "web"),
            public_key_pem="-----BEGIN PUBLIC KEY-----\nMIIBIjANBg...",
            validity_days=7,
        )
        
        cert = pki.issue_certificate(csr)
        assert pki.validate_certificate(cert)
    
    def test_rotate_certificate(self):
        """Test certificate rotation."""
        pki = PKIManager()
        
        # Issue initial cert
        csr = CertificateSigningRequest(
            spiffe_id=create_spiffe_id("service", "auth"),
            public_key_pem="-----BEGIN PUBLIC KEY-----\nMIIBIjANBg...",
            validity_days=7,
        )
        
        old_cert = pki.issue_certificate(csr)
        
        # Rotate to new cert
        new_cert = pki.rotate_certificate(old_cert)
        
        assert old_cert.state == CertificateState.ROTATING
        assert new_cert.state == CertificateState.VALID
        assert old_cert.specs.spiffe_id == new_cert.specs.spiffe_id


class TestPolicy:
    """Test authorization policy entities."""
    
    def test_principal_roles(self):
        """Test principal with roles."""
        principal = Principal(
            principal_id=PrincipalId("svc-1"),
            spiffe_id=create_spiffe_id("service", "reader"),
            principal_type="service",
            roles=["reader", "auditor"],
        )
        
        assert principal.has_role("reader")
        assert principal.has_role("auditor")
        assert not principal.has_role("writer")
    
    def test_resource_tags(self):
        """Test resource with tags."""
        resource = Resource(
            resource_id=ResourceId("database/users"),
            resource_type="database",
            tags={"classification": "confidential", "env": "production"},
        )
        
        assert resource.matches_tag("classification", "confidential")
        assert not resource.matches_tag("classification", "public")


class TestPolicyEngine:
    """Test authorization policy evaluation."""
    
    def test_rbac_allow(self):
        """Test RBAC-based allow decision."""
        engine = PolicyEngine()
        
        # Create RBAC policy
        rule = PolicyRule(
            rule_id="read-rule",
            policy_type=PolicyType.RBAC,
            effect=Effect.ALLOW,
            roles=["reader"],
            actions=["read"],
        )
        
        policy = Policy(
            policy_id=create_policy_id("reader_policy"),
            policy_name="Reader Policy",
            policy_type=PolicyType.RBAC,
            rules=[rule],
        )
        
        engine.add_policy(policy)
        
        # Evaluate request
        principal = Principal(
            principal_id=PrincipalId("svc-1"),
            spiffe_id=create_spiffe_id("service", "reader"),
            principal_type="service",
            roles=["reader"],
        )
        
        resource = Resource(
            resource_id=ResourceId("database/users"),
            resource_type="database",
        )
        
        decision = engine.evaluate(principal, "read", resource)
        assert decision.allowed
    
    def test_default_deny(self):
        """Test default deny when no policy matches."""
        engine = PolicyEngine()
        
        principal = Principal(
            principal_id=PrincipalId("svc-1"),
            spiffe_id=create_spiffe_id("service", "guest"),
            principal_type="service",
            roles=[],
        )
        
        resource = Resource(
            resource_id=ResourceId("database/secrets"),
            resource_type="database",
        )
        
        decision = engine.evaluate(principal, "delete", resource)
        assert not decision.allowed


class TestCoordinator:
    """Test SecureCoordinator integration."""

    def test_coordinator_certificate_flow(self):
        """Test certificate request through coordinator."""
        from secure_platform.application.coordinator import SecureCoordinator

        coordinator = SecureCoordinator()
        coordinator.initialize()

        cert = coordinator.request_certificate(
            requester="alice",
            spiffe_id=create_spiffe_id("service", "payment"),
            public_key_pem="-----BEGIN PUBLIC KEY-----\nMIIBIjANBg...",
            validity_days=7,
        )

        assert cert is not None
        assert cert.is_valid
        assert "payment" in cert.specs.spiffe_id

        # Check metrics updated
        metrics = coordinator.get_metrics()
        assert metrics["certs_issued"] >= 1

    def test_coordinator_authorization_flow(self):
        """Test authorization through coordinator with Resource object."""
        from secure_platform.application.coordinator import SecureCoordinator

        coordinator = SecureCoordinator()
        coordinator.initialize()

        # Create a policy first
        rule = PolicyRule(
            rule_id="admin-read-rule",
            policy_type=PolicyType.RBAC,
            effect=Effect.ALLOW,
            roles=["admin"],
            actions=["read", "write"],
        )

        policy = Policy(
            policy_id=create_policy_id("admin_policy"),
            policy_name="Admin Policy",
            policy_type=PolicyType.RBAC,
            rules=[rule],
        )

        coordinator.create_policy(creator="system", policy=policy)

        # Build principal with admin role
        principal = Principal(
            principal_id=PrincipalId("admin-1"),
            spiffe_id=create_spiffe_id("user", "admin"),
            principal_type="user",
            roles=["admin"],
        )

        # Build resource
        resource = Resource(
            resource_id=ResourceId("database/users"),
            resource_type="database",
            attributes={"classification": "internal"},
        )

        # Authorize - now passes Resource object (not string)
        decision = coordinator.authorize_request(
            principal=principal,
            action="read",
            resource=resource,
        )

        assert decision.allowed
        assert "admin-read-rule" in decision.reason

        # Check metrics
        metrics = coordinator.get_metrics()
        assert metrics["auth_allowed"] >= 1

    def test_coordinator_deny_flow(self):
        """Test authorization denial through coordinator."""
        from secure_platform.application.coordinator import SecureCoordinator

        coordinator = SecureCoordinator()
        coordinator.initialize()

        # Principal without any roles
        principal = Principal(
            principal_id=PrincipalId("guest-1"),
            spiffe_id=create_spiffe_id("user", "guest"),
            principal_type="user",
            roles=[],
        )

        resource = Resource(
            resource_id=ResourceId("database/secrets"),
            resource_type="database",
        )

        # Should be denied (default deny, no matching policy)
        decision = coordinator.authorize_request(
            principal=principal,
            action="delete",
            resource=resource,
        )

        assert not decision.allowed
        assert "default deny" in decision.reason.lower()

        # Check metrics
        metrics = coordinator.get_metrics()
        assert metrics["auth_denied"] >= 1


class TestAudit:
    """Test audit logging with Merkle tree."""
    
    def test_event_hashing(self):
        """Test audit event hashing."""
        event = AuditEvent(
            event_id="evt-1",
            event_type=AuditEventType.CERT_ISSUED,
            principal_id="admin",
            action="issue",
            resource_id="spiffe://cluster/service/api",
            result="success",
        )
        
        hash1 = event.hash()
        hash2 = event.hash()
        
        assert hash1 == hash2  # Deterministic
        assert len(hash1) == 64  # SHA-256 hex
    
    def test_merkle_tree_append(self):
        """Test appending events to Merkle tree."""
        tree = MerkleTree()
        
        for i in range(5):
            event = AuditEvent(
                event_id=f"evt-{i}",
                event_type=AuditEventType.AUTH_ALLOWED,
                principal_id="svc-1",
                action="read",
                resource_id="database/users",
                result="success",
            )
            tree.append(event)
        
        # Verify root hash changed
        assert tree.get_root_hash() is not None
    
    def test_merkle_proof_verification(self):
        """Test Merkle proof verification."""
        tree = MerkleTree()
        
        # Add events
        events = []
        for i in range(4):
            event = AuditEvent(
                event_id=f"evt-{i}",
                event_type=AuditEventType.AUTH_ALLOWED,
                principal_id="svc-1",
                action="read",
                resource_id=f"database/table{i}",
                result="success",
            )
            tree.append(event)
            events.append(event)
        
        # Get proof for event 2
        proof = tree.get_proof("evt-2")
        root_hash = tree.get_root_hash()
        
        assert proof is not None
        assert tree.verify_proof(proof, root_hash)
    
    def test_audit_logger(self):
        """Test audit logger."""
        logger = AuditLogger()
        
        # Log some events
        event1 = logger.log_cert_issued("admin", "cert-123", "spiffe://cluster/service/api")
        event2 = logger.log_auth_allowed("svc-1", "read", "database/users")
        event3 = logger.log_auth_denied("guest", "write", "database/secrets", reason="No write permission")
        
        # Verify events are logged
        assert logger.get_event(event1.event_id) is not None
        assert logger.get_event(event2.event_id) is not None
        assert logger.get_event(event3.event_id) is not None
        
        # Verify events are in Merkle tree (root hash exists)
        assert logger.get_root_hash() is not None
        
        # Verify stats
        stats = logger.get_stats()
        assert stats['total_events'] == 3
        assert stats['events_by_type']['cert_issued'] == 1
        assert stats['events_by_type']['auth_allowed'] == 1
        assert stats['events_by_type']['auth_denied'] == 1
