"""Authorization policy engine (RBAC + ABAC).

Evaluates access control policies using role-based and attribute-based rules.

References:
    - design.md Section 3 (Policy Engine)
    - NIST SP 800-162 (ABAC Guide)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from secure_platform.domain.entities.policy import (
    AuthorizationDecision,
    Effect,
    Policy,
    PolicyRule,
    PolicyType,
    Principal,
    Resource,
)

logger = logging.getLogger(__name__)


class PolicyEngineError(Exception):
    """Policy engine error."""
    pass


class PolicyEngine:
    """Evaluate authorization policies."""
    
    def __init__(self):
        """Initialize policy engine."""
        self._policies: dict[str, Policy] = {}
        self._decision_cache: dict[str, AuthorizationDecision] = {}
    
    def add_policy(self, policy: Policy) -> None:
        """Add an authorization policy.
        
        Args:
            policy: Policy to add.
        """
        self._policies[str(policy.policy_id)] = policy
        logger.info(f"Added policy {policy.policy_name} ({policy.policy_type.value})")
    
    def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy.
        
        Args:
            policy_id: Policy ID to remove.
            
        Returns:
            True if policy was removed.
        """
        if policy_id in self._policies:
            del self._policies[policy_id]
            logger.info(f"Removed policy {policy_id}")
            return True
        return False
    
    def evaluate(
        self,
        principal: Principal,
        action: str,
        resource: Resource,
    ) -> AuthorizationDecision:
        """Evaluate if principal can perform action on resource.
        
        Uses default DENY policy: explicit ALLOW required.
        
        Args:
            principal: Principal making the request.
            action: Action to perform (e.g., "read", "write").
            resource: Resource being accessed.
            
        Returns:
            AuthorizationDecision with allow/deny.
        """
        start_time = time.time()
        
        # Check policies in order (first match wins)
        for policy in self._policies.values():
            if not policy.enabled:
                continue
            
            for rule in policy.rules:
                if self._rule_matches(rule, principal, action, resource):
                    decision_time = (time.time() - start_time) * 1000
                    
                    if rule.effect == Effect.ALLOW:
                        return AuthorizationDecision(
                            allowed=True,
                            reason=f"Allowed by rule {rule.rule_id}",
                            matching_rule_id=rule.rule_id,
                            decision_time_ms=decision_time,
                        )
                    else:  # DENY
                        return AuthorizationDecision(
                            allowed=False,
                            reason=f"Denied by rule {rule.rule_id}",
                            matching_rule_id=rule.rule_id,
                            decision_time_ms=decision_time,
                        )
        
        # Default DENY
        decision_time = (time.time() - start_time) * 1000
        return AuthorizationDecision(
            allowed=False,
            reason="No matching allow policy (default deny)",
            decision_time_ms=decision_time,
        )
    
    def _rule_matches(self, rule: PolicyRule, principal: Principal, action: str, resource: Resource) -> bool:
        """Check if a rule matches the request.
        
        Args:
            rule: Policy rule to check.
            principal: Principal making request.
            action: Action being requested.
            resource: Resource being accessed.
            
        Returns:
            True if rule matches.
        """
        # Check action
        if rule.actions and action not in rule.actions and "*" not in rule.actions:
            return False
        
        # Check resource (wildcard matching)
        if rule.resources:
            if not self._resource_matches(action, resource, rule.resources):
                return False
        
        # Type-specific checks
        if rule.policy_type == PolicyType.RBAC:
            return self._rbac_matches(rule, principal)
        
        elif rule.policy_type == PolicyType.ABAC:
            return self._abac_matches(rule, principal, resource)
        
        elif rule.policy_type == PolicyType.ACL:
            return self._acl_matches(rule, principal)
        
        return False
    
    def _rbac_matches(self, rule: PolicyRule, principal: Principal) -> bool:
        """Check if RBAC rule matches.
        
        Args:
            rule: RBAC rule.
            principal: Principal to check.
            
        Returns:
            True if principal has required role.
        """
        if not rule.roles:
            return True
        
        return any(principal.has_role(role) for role in rule.roles)
    
    def _abac_matches(self, rule: PolicyRule, principal: Principal, resource: Resource) -> bool:
        """Check if ABAC rule matches.
        
        Args:
            rule: ABAC rule with conditions.
            principal: Principal to check.
            resource: Resource to check.
            
        Returns:
            True if all conditions match.
        """
        for condition_key, expected_value in rule.conditions.items():
            # Check principal attributes
            if condition_key.startswith("principal."):
                attr_name = condition_key[len("principal."):]
                principal_value = principal.get_attribute(attr_name)
                if principal_value != expected_value:
                    return False
            
            # Check resource attributes
            elif condition_key.startswith("resource."):
                attr_name = condition_key[len("resource."):]
                resource_value = resource.attributes.get(attr_name)
                if resource_value != expected_value:
                    return False
        
        return True
    
    def _acl_matches(self, rule: PolicyRule, principal: Principal) -> bool:
        """Check if ACL rule matches.
        
        Args:
            rule: ACL rule.
            principal: Principal to check.
            
        Returns:
            True if principal is in ACL.
        """
        if not rule.principals:
            return True
        
        principal_str = str(principal.principal_id)
        return principal_str in rule.principals
    
    def _resource_matches(self, action: str, resource: Resource, patterns: list[str]) -> bool:
        """Check if resource matches pattern list.
        
        Supports wildcards: "database/*", "bucket/confidential"
        
        Args:
            action: Action being performed.
            resource: Resource to check.
            patterns: Resource patterns to match.
            
        Returns:
            True if resource matches any pattern.
        """
        resource_str = str(resource.resource_id)
        
        for pattern in patterns:
            if "*" in pattern:
                # Simple wildcard matching
                pattern_parts = pattern.split("*")
                if all(part in resource_str for part in pattern_parts):
                    return True
            else:
                if resource_str == pattern:
                    return True
        
        return False
    
    def get_stats(self) -> dict:
        """Get policy engine statistics.
        
        Returns:
            Dictionary with stats.
        """
        total_rules = sum(len(p.rules) for p in self._policies.values())
        
        return {
            'total_policies': len(self._policies),
            'enabled_policies': sum(1 for p in self._policies.values() if p.enabled),
            'total_rules': total_rules,
        }
