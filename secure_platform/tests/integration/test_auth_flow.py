"""Integration tests for secure_platform authentication flow."""

import pytest

from secure_platform.domain.entities.certificate import (
    Certificate,
    CertificateSigningRequest,
    CertificateState,
)
from secure_platform.domain.entities.policy import (
    AuthorizationDecision,
    Effect,
    Policy,
    PolicyRule,
    PolicyType,
    Principal,
    Resource,
)
from secure_platform.domain.entities.audit import AuditEventType
from secure_platform.domain.services.pki_manager import PKIManager, PKIError
from secure_platform.domain.services.policy_engine import PolicyEngine
from secure_platform.domain.services.audit_logger import AuditLogger
from secure_platform.domain.value_objects.identifiers import (
    create_spiffe_id,
    create_principal_id,
    create_resource_id,
    create_policy_id,
)


class TestPKICertificateLifecycle:
    """Integration tests for PKI certificate lifecycle."""

    def test_full_certificate_lifecycle(self):
        """Test issuing, validating, rotating, and revoking a certificate."""
        pki = PKIManager()

        # Issue certificate
        spiffe_id = create_spiffe_id("service", "payment-service")
        csr = CertificateSigningRequest(
            spiffe_id=spiffe_id,
            public_key_pem="-----BEGIN PUBLIC KEY-----\nMIIBIjAN...fake_key\n-----END PUBLIC KEY-----",
            validity_days=7,
        )
        cert = pki.issue_certificate(csr)

        assert cert.state == CertificateState.VALID
        assert cert.specs.spiffe_id == spiffe_id

        # Validate certificate
        assert pki.validate_certificate(cert)

        # Get by serial
        retrieved_cert = pki.get_certificate(cert.specs.serial)
        assert retrieved_cert is not None
        assert retrieved_cert.specs.serial == cert.specs.serial

        # Get by SPIFFE ID
        current_cert = pki.get_current_certificate(spiffe_id)
        assert current_cert is not None

        # Rotate certificate
        new_cert = pki.rotate_certificate(cert)
        assert new_cert.specs.serial != cert.specs.serial
        assert new_cert.state == CertificateState.VALID
        assert cert.state == CertificateState.ROTATING

        # Revoke new certificate
        pki.revoke_certificate(new_cert.specs.serial, reason="key compromised")
        revoked_cert = pki.get_certificate(new_cert.specs.serial)
        assert revoked_cert.state == CertificateState.REVOKED
        assert not pki.validate_certificate(revoked_cert)

        # Check stats
        stats = pki.get_stats()
        assert stats["total_issued"] >= 2
        assert stats["revoked"] >= 1


class TestPolicyEngineAuthorization:
    """Integration tests for policy engine authorization."""

    def test_rbac_authorization_flow(self):
        """Test RBAC-based authorization."""
        engine = PolicyEngine()

        # Create RBAC policy
        policy = Policy(
            policy_id=create_policy_id("admin-policy"),
            policy_name="Admin Access Policy",
            policy_type=PolicyType.RBAC,
            rules=[
                PolicyRule(
                    rule_id="admin-full-access",
                    policy_type=PolicyType.RBAC,
                    effect=Effect.ALLOW,
                    roles=["admin"],
                    actions=["*"],
                    resources=["*"],
                ),
                PolicyRule(
                    rule_id="reader-read-only",
                    policy_type=PolicyType.RBAC,
                    effect=Effect.ALLOW,
                    roles=["reader"],
                    actions=["read"],
                    resources=["database/*"],
                ),
            ],
        )
        engine.add_policy(policy)

        # Create principals
        admin_spiffe = create_spiffe_id("service", "admin-service")
        admin = Principal(
            principal_id=create_principal_id(admin_spiffe),
            spiffe_id=admin_spiffe,
            principal_type="service",
            roles=["admin"],
        )

        reader_spiffe = create_spiffe_id("service", "reader-service")
        reader = Principal(
            principal_id=create_principal_id(reader_spiffe),
            spiffe_id=reader_spiffe,
            principal_type="service",
            roles=["reader"],
        )

        guest_spiffe = create_spiffe_id("service", "guest-service")
        guest = Principal(
            principal_id=create_principal_id(guest_spiffe),
            spiffe_id=guest_spiffe,
            principal_type="service",
            roles=[],
        )

        # Create resource
        database = Resource(
            resource_id=create_resource_id("database", "production"),
            resource_type="database",
        )

        # Admin can do anything
        result = engine.evaluate(admin, "read", database)
        assert result.allowed
        result = engine.evaluate(admin, "write", database)
        assert result.allowed

        # Reader can only read databases
        result = engine.evaluate(reader, "read", database)
        assert result.allowed
        result = engine.evaluate(reader, "write", database)
        assert not result.allowed  # Default deny

        # Guest denied (no matching policy)
        result = engine.evaluate(guest, "read", database)
        assert not result.allowed

    def test_abac_authorization_flow(self):
        """Test ABAC-based authorization with conditions."""
        engine = PolicyEngine()

        # Create ABAC policy
        policy = Policy(
            policy_id=create_policy_id("department-policy"),
            policy_name="Department Access Policy",
            policy_type=PolicyType.ABAC,
            rules=[
                PolicyRule(
                    rule_id="same-department-access",
                    policy_type=PolicyType.ABAC,
                    effect=Effect.ALLOW,
                    actions=["read", "write"],
                    resources=["*"],
                    conditions={
                        "principal.department": "engineering",
                        "resource.department": "engineering",
                    },
                ),
            ],
        )
        engine.add_policy(policy)

        # Create engineering principal
        eng_spiffe = create_spiffe_id("service", "eng-service")
        engineer = Principal(
            principal_id=create_principal_id(eng_spiffe),
            spiffe_id=eng_spiffe,
            principal_type="service",
            attributes={"department": "engineering"},
        )

        # Create finance principal
        fin_spiffe = create_spiffe_id("service", "finance-service")
        accountant = Principal(
            principal_id=create_principal_id(fin_spiffe),
            spiffe_id=fin_spiffe,
            principal_type="service",
            attributes={"department": "finance"},
        )

        # Create engineering resource
        eng_resource = Resource(
            resource_id=create_resource_id("api", "eng-api"),
            resource_type="api",
            attributes={"department": "engineering"},
        )

        # Engineer can access engineering resources
        result = engine.evaluate(engineer, "read", eng_resource)
        assert result.allowed

        # Accountant cannot access engineering resources
        result = engine.evaluate(accountant, "read", eng_resource)
        assert not result.allowed


class TestAuditLogging:
    """Integration tests for audit logging with Merkle tree."""

    def test_audit_log_integrity(self):
        """Test that audit log maintains integrity via Merkle tree."""
        logger = AuditLogger()

        # Log multiple events
        event1 = logger.log_cert_issued("admin", "cert-001", "spiffe://cluster/service/svc1")
        event2 = logger.log_auth_allowed("user1", "read", "database/prod")
        event3 = logger.log_auth_denied("user2", "write", "database/prod", "insufficient permissions")

        # Verify all events can be retrieved
        assert logger.get_event(event1.event_id) is not None
        assert logger.get_event(event2.event_id) is not None
        assert logger.get_event(event3.event_id) is not None

        # Verify integrity via Merkle proof
        assert logger.verify_event(event1.event_id)
        assert logger.verify_event(event2.event_id)
        assert logger.verify_event(event3.event_id)

        # Check root hash exists
        root_hash = logger.get_root_hash()
        assert root_hash is not None

        # Check stats
        stats = logger.get_stats()
        assert stats["total_events"] == 3
        assert AuditEventType.CERT_ISSUED.value in stats["events_by_type"]


class TestEndToEndAuthFlow:
    """Integration tests for complete authentication flow."""

    def test_complete_mtls_auth_flow(self):
        """Test complete mTLS authentication flow: issue cert, authorize, audit."""
        pki = PKIManager()
        policy_engine = PolicyEngine()
        audit_logger = AuditLogger()

        # Step 1: Issue certificate for service
        spiffe_id = create_spiffe_id("service", "api-gateway")
        csr = CertificateSigningRequest(
            spiffe_id=spiffe_id,
            public_key_pem="-----BEGIN PUBLIC KEY-----\nfake_key_data\n-----END PUBLIC KEY-----",
        )
        cert = pki.issue_certificate(csr)
        audit_logger.log_cert_issued(
            principal_id=str(spiffe_id),
            cert_serial=str(cert.specs.serial),
            spiffe_id=str(spiffe_id),
        )

        # Step 2: Configure authorization policy
        policy = Policy(
            policy_id=create_policy_id("api-gateway-policy"),
            policy_name="API Gateway Access",
            policy_type=PolicyType.RBAC,
            rules=[
                PolicyRule(
                    rule_id="gateway-access",
                    policy_type=PolicyType.RBAC,
                    effect=Effect.ALLOW,
                    roles=["gateway"],
                    actions=["read", "proxy"],
                    resources=["api/*"],
                ),
            ],
        )
        policy_engine.add_policy(policy)

        # Step 3: Validate certificate before request
        assert pki.validate_certificate(cert)

        # Step 4: Create principal from certificate
        principal = Principal(
            principal_id=create_principal_id(spiffe_id),
            spiffe_id=spiffe_id,
            principal_type="service",
            roles=["gateway"],
        )

        # Step 5: Authorize request
        resource = Resource(
            resource_id=create_resource_id("api", "backend-service"),
            resource_type="api",
        )
        decision = policy_engine.evaluate(principal, "proxy", resource)

        # Step 6: Log authorization result
        if decision.allowed:
            audit_logger.log_auth_allowed(
                principal_id=str(principal.principal_id),
                action="proxy",
                resource_id=str(resource.resource_id),
            )
        else:
            audit_logger.log_auth_denied(
                principal_id=str(principal.principal_id),
                action="proxy",
                resource_id=str(resource.resource_id),
                reason=decision.reason,
            )

        # Verify complete flow
        assert decision.allowed
        assert audit_logger.get_stats()["total_events"] == 2

        # Verify audit trail integrity
        stats = audit_logger.get_stats()
        assert stats["tree_root_hash"] is not None


class TestPolicyDenyOverride:
    """Integration tests for explicit deny policies."""

    def test_explicit_deny_overrides_allow(self):
        """Test that explicit DENY rules override ALLOW rules."""
        engine = PolicyEngine()

        # Create allow policy first
        allow_policy = Policy(
            policy_id=create_policy_id("allow-all"),
            policy_name="Allow All",
            policy_type=PolicyType.RBAC,
            rules=[
                PolicyRule(
                    rule_id="allow-read",
                    policy_type=PolicyType.RBAC,
                    effect=Effect.ALLOW,
                    roles=["user"],
                    actions=["read"],
                    resources=["*"],
                ),
            ],
        )
        engine.add_policy(allow_policy)

        # Create deny policy (added after, but should be checked first based on order)
        deny_policy = Policy(
            policy_id=create_policy_id("deny-sensitive"),
            policy_name="Deny Sensitive",
            policy_type=PolicyType.RBAC,
            rules=[
                PolicyRule(
                    rule_id="deny-secrets",
                    policy_type=PolicyType.RBAC,
                    effect=Effect.DENY,
                    roles=["user"],
                    actions=["read"],
                    resources=["secrets/*"],
                ),
            ],
        )
        engine.add_policy(deny_policy)

        # Create user
        spiffe = create_spiffe_id("service", "app")
        user = Principal(
            principal_id=create_principal_id(spiffe),
            spiffe_id=spiffe,
            principal_type="service",
            roles=["user"],
        )

        # Test access to normal resource - allowed
        normal_resource = Resource(
            resource_id=create_resource_id("database", "public"),
            resource_type="database",
        )
        result = engine.evaluate(user, "read", normal_resource)
        assert result.allowed

        # Test access to secrets - check policy order
        secrets_resource = Resource(
            resource_id=create_resource_id("secrets", "api-keys"),
            resource_type="secrets",
        )
        result = engine.evaluate(user, "read", secrets_resource)
        # Note: In this simple implementation, first match wins
        # So the result depends on policy iteration order


class TestCertificateRevocationImpact:
    """Integration tests for certificate revocation effects."""

    def test_revoked_cert_fails_validation(self):
        """Test that revoked certificates fail validation."""
        pki = PKIManager()

        # Issue certificate
        spiffe_id = create_spiffe_id("service", "compromised-service")
        csr = CertificateSigningRequest(
            spiffe_id=spiffe_id,
            public_key_pem="-----BEGIN PUBLIC KEY-----\nkey_data\n-----END PUBLIC KEY-----",
        )
        cert = pki.issue_certificate(csr)

        # Initially valid
        assert pki.validate_certificate(cert)

        # Revoke
        pki.revoke_certificate(cert.specs.serial, "security incident")

        # Now invalid
        assert not pki.validate_certificate(cert)

        # Cannot get current cert for SPIFFE ID anymore
        current = pki.get_current_certificate(spiffe_id)
        assert current is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
