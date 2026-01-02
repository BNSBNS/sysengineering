"""FastAPI REST adapter for the Secure Platform.

Provides HTTP endpoints for certificate management, authorization, and audit.

Usage:
    from secure_platform.adapters.inbound.rest_api import create_app

    coordinator = SecureCoordinator()
    coordinator.initialize()
    app = create_app(coordinator)
    # Run with: uvicorn module:app --host 0.0.0.0 --port 8443

References:
    - design.md Section 2.4 (API Services)
    - ports/inbound/api.py (API contracts)
"""

from __future__ import annotations

from typing import Optional

try:
    from fastapi import FastAPI, HTTPException, status
    from pydantic import BaseModel, Field

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


# Pydantic models for request/response serialization
if _HAS_FASTAPI:

    class IssueCertificateRequest(BaseModel):
        """Request to issue a new certificate."""

        requester: str = Field(..., min_length=1, description="Who is requesting")
        spiffe_id: str = Field(..., min_length=1, description="SPIFFE ID for cert")
        public_key_pem: str = Field(..., min_length=1, description="Public key PEM")
        validity_days: int = Field(default=7, ge=1, le=365, description="Validity days")

    class CertificateResponse(BaseModel):
        """Certificate details response."""

        serial: str
        spiffe_id: str
        state: str
        valid_from: float
        valid_until: float
        public_key_pem: str

    class AuthorizeRequestModel(BaseModel):
        """Authorization request."""

        principal_id: str = Field(..., description="Principal making the request")
        principal_type: str = Field(default="service", description="Type: service, user, node")
        spiffe_id: str = Field(..., description="Principal's SPIFFE ID")
        roles: list[str] = Field(default_factory=list, description="Principal's roles")
        attributes: dict = Field(default_factory=dict, description="Principal attributes")
        action: str = Field(..., description="Action to perform")
        resource_id: str = Field(..., description="Resource to access")
        resource_type: str = Field(default="api", description="Resource type")
        resource_attributes: dict = Field(default_factory=dict, description="Resource attributes")

    class AuthorizationResponse(BaseModel):
        """Authorization decision response."""

        allowed: bool
        reason: str
        matching_rule_id: Optional[str] = None
        decision_time_ms: float

    class PolicyRuleModel(BaseModel):
        """Single policy rule."""

        rule_id: str
        policy_type: str = Field(..., description="rbac, abac, or acl")
        effect: str = Field(..., description="allow or deny")
        description: str = ""
        roles: list[str] = Field(default_factory=list)
        conditions: dict = Field(default_factory=dict)
        principals: list[str] = Field(default_factory=list)
        actions: list[str] = Field(default_factory=list)
        resources: list[str] = Field(default_factory=list)

    class CreatePolicyRequest(BaseModel):
        """Request to create a policy."""

        creator: str = Field(..., description="Who is creating the policy")
        policy_id: str = Field(..., description="Unique policy ID")
        policy_name: str = Field(..., description="Policy name")
        policy_type: str = Field(..., description="rbac, abac, or acl")
        rules: list[PolicyRuleModel] = Field(default_factory=list)
        enabled: bool = True

    class AuditStatsResponse(BaseModel):
        """Audit statistics response."""

        total_events: int
        events_by_type: dict
        tree_root_hash: Optional[str]

    class MetricsResponse(BaseModel):
        """Platform metrics response."""

        certs_issued: int
        certs_rotated: int
        certs_revoked: int
        auth_allowed: int
        auth_denied: int
        policies_created: int
        audit_stats: dict
        audit_root_hash: Optional[str]

    class HealthResponse(BaseModel):
        """Health check response."""

        status: str
        version: str = "0.1.0"


def create_app(coordinator) -> "FastAPI":
    """Create FastAPI application with Secure Platform endpoints.

    Args:
        coordinator: SecureCoordinator instance.

    Returns:
        Configured FastAPI application.

    Raises:
        ImportError: If FastAPI is not installed.
    """
    if not _HAS_FASTAPI:
        raise ImportError(
            "FastAPI not installed. Install with: pip install fastapi uvicorn"
        )

    from secure_platform.domain.entities.policy import (
        Effect,
        Policy,
        PolicyRule,
        PolicyType,
        Principal,
        Resource,
    )

    app = FastAPI(
        title="Secure Platform API",
        description="Zero Trust security control plane with PKI, RBAC/ABAC authorization, and tamper-evident audit",
        version="1.0.0",
    )

    # Type mappings
    POLICY_TYPE_MAP = {
        "rbac": PolicyType.RBAC,
        "abac": PolicyType.ABAC,
        "acl": PolicyType.ACL,
    }
    EFFECT_MAP = {
        "allow": Effect.ALLOW,
        "deny": Effect.DENY,
    }

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Check platform health status."""
        return HealthResponse(status="healthy")

    @app.get("/metrics", response_model=MetricsResponse, tags=["System"])
    async def get_metrics():
        """Get platform metrics."""
        metrics = coordinator.get_metrics()
        return MetricsResponse(**metrics)

    @app.post(
        "/certificates",
        response_model=CertificateResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Certificates"],
    )
    async def issue_certificate(request: IssueCertificateRequest):
        """Issue a new X.509 certificate."""
        try:
            cert = coordinator.request_certificate(
                requester=request.requester,
                spiffe_id=request.spiffe_id,
                public_key_pem=request.public_key_pem,
                validity_days=request.validity_days,
            )
            return CertificateResponse(
                serial=cert.specs.serial,
                spiffe_id=cert.specs.spiffe_id,
                state=cert.state.value,
                valid_from=cert.specs.valid_from.timestamp(),
                valid_until=cert.specs.valid_until.timestamp(),
                public_key_pem=cert.specs.public_key_pem,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

    @app.post("/authorize", response_model=AuthorizationResponse, tags=["Authorization"])
    async def authorize_request(request: AuthorizeRequestModel):
        """Authorize an access request using RBAC/ABAC policies."""
        # Build Principal
        principal = Principal(
            principal_id=request.principal_id,
            spiffe_id=request.spiffe_id,
            principal_type=request.principal_type,
            roles=request.roles,
            attributes=request.attributes,
        )

        # Build Resource
        resource = Resource(
            resource_id=request.resource_id,
            resource_type=request.resource_type,
            attributes=request.resource_attributes,
        )

        try:
            decision = coordinator.authorize_request(
                principal=principal,
                action=request.action,
                resource=resource,
            )
            return AuthorizationResponse(
                allowed=decision.allowed,
                reason=decision.reason,
                matching_rule_id=decision.matching_rule_id,
                decision_time_ms=decision.decision_time_ms,
            )
        except RuntimeError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e),
            )

    @app.post(
        "/policies",
        response_model=dict,
        status_code=status.HTTP_201_CREATED,
        tags=["Policies"],
    )
    async def create_policy(request: CreatePolicyRequest):
        """Create a new authorization policy."""
        policy_type = POLICY_TYPE_MAP.get(request.policy_type.lower())
        if not policy_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid policy_type: {request.policy_type}. Must be: rbac, abac, acl",
            )

        # Convert rules
        rules = []
        for r in request.rules:
            rule_type = POLICY_TYPE_MAP.get(r.policy_type.lower())
            if not rule_type:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid rule policy_type: {r.policy_type}",
                )
            effect = EFFECT_MAP.get(r.effect.lower())
            if not effect:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid effect: {r.effect}. Must be: allow, deny",
                )

            rules.append(
                PolicyRule(
                    rule_id=r.rule_id,
                    policy_type=rule_type,
                    effect=effect,
                    description=r.description,
                    roles=r.roles,
                    conditions=r.conditions,
                    principals=r.principals,
                    actions=r.actions,
                    resources=r.resources,
                )
            )

        policy = Policy(
            policy_id=request.policy_id,
            policy_name=request.policy_name,
            policy_type=policy_type,
            rules=rules,
            enabled=request.enabled,
        )

        try:
            coordinator.create_policy(creator=request.creator, policy=policy)
            return {"policy_id": request.policy_id, "status": "created"}
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

    @app.get("/audit/stats", response_model=AuditStatsResponse, tags=["Audit"])
    async def get_audit_stats():
        """Get audit log statistics."""
        stats = coordinator.get_audit_stats()
        return AuditStatsResponse(**stats)

    @app.get("/audit/root-hash", response_model=dict, tags=["Audit"])
    async def get_audit_root_hash():
        """Get audit log Merkle tree root hash for integrity verification."""
        metrics = coordinator.get_metrics()
        return {"root_hash": metrics.get("audit_root_hash")}

    return app
