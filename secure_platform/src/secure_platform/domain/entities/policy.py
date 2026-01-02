"""Authorization policy and principal entities.

Implements RBAC (Role-Based Access Control) and ABAC (Attribute-Based Access Control)
for fine-grained access control decisions.

References:
    - design.md Section 3 (Policy Engine)
    - NIST SP 800-162 (ABAC Guide)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from secure_platform.domain.value_objects.identifiers import (
    PolicyId,
    PrincipalId,
    ResourceId,
    SpiffeId,
)


class PolicyType(Enum):
    """Type of authorization policy."""
    RBAC = "rbac"  # Role-based (principal has role, role has permission)
    ABAC = "abac"  # Attribute-based (conditions on attributes)
    ACL = "acl"    # Access control list (explicit allow/deny)


class Effect(Enum):
    """Effect of a policy rule."""
    ALLOW = "allow"
    DENY = "deny"


@dataclass
class Principal:
    """Security principal (service, user, etc.)."""
    principal_id: PrincipalId
    spiffe_id: SpiffeId
    principal_type: str  # "service", "user", "node", etc.
    roles: list[str] = field(default_factory=list)  # RBAC roles
    attributes: dict[str, Any] = field(default_factory=dict)  # ABAC attributes
    
    def has_role(self, role: str) -> bool:
        """Check if principal has a role.
        
        Args:
            role: Role name to check.
            
        Returns:
            True if principal has the role.
        """
        return role in self.roles
    
    def get_attribute(self, attr_name: str) -> Optional[Any]:
        """Get an attribute value.
        
        Args:
            attr_name: Attribute name.
            
        Returns:
            Attribute value or None if not found.
        """
        return self.attributes.get(attr_name)


@dataclass
class Resource:
    """Protected resource."""
    resource_id: ResourceId
    resource_type: str  # "database", "bucket", "api", etc.
    owner: Optional[PrincipalId] = None
    attributes: dict[str, Any] = field(default_factory=dict)  # ABAC attributes
    tags: dict[str, str] = field(default_factory=dict)  # Classification tags
    
    def matches_tag(self, tag_key: str, tag_value: str) -> bool:
        """Check if resource has a tag.
        
        Args:
            tag_key: Tag key.
            tag_value: Tag value.
            
        Returns:
            True if resource has matching tag.
        """
        return self.tags.get(tag_key) == tag_value


@dataclass
class Role:
    """RBAC role with associated permissions."""
    role_name: str
    description: str = ""
    permissions: list[str] = field(default_factory=list)  # e.g., ["read", "write"]
    
    def can_perform(self, action: str) -> bool:
        """Check if role grants permission for action.
        
        Args:
            action: Action to check (e.g., "read", "write").
            
        Returns:
            True if role has permission.
        """
        return action in self.permissions


@dataclass
class PolicyRule:
    """Single authorization policy rule."""
    rule_id: str
    policy_type: PolicyType
    effect: Effect
    description: str = ""
    
    # For RBAC
    roles: list[str] = field(default_factory=list)  # Roles that match
    
    # For ABAC
    conditions: dict[str, Any] = field(default_factory=dict)  # Attribute conditions
    
    # For ACL
    principals: list[str] = field(default_factory=list)  # Explicit principals
    
    # Action and resource constraints
    actions: list[str] = field(default_factory=list)  # e.g., ["read", "write", "*"]
    resources: list[str] = field(default_factory=list)  # e.g., ["database/*", "bucket/confidential"]


@dataclass
class Policy:
    """Authorization policy with multiple rules."""
    policy_id: PolicyId
    policy_name: str
    policy_type: PolicyType
    rules: list[PolicyRule] = field(default_factory=list)
    enabled: bool = True
    created_timestamp: float = field(default_factory=lambda: __import__('time').time())


@dataclass
class AuthorizationDecision:
    """Result of an authorization check."""
    allowed: bool
    reason: str  # Why it was allowed/denied
    matching_rule_id: Optional[str] = None  # Which rule matched
    decision_time_ms: float = 0.0  # Decision latency
    evaluated_at_timestamp: float = field(default_factory=lambda: __import__('time').time())
