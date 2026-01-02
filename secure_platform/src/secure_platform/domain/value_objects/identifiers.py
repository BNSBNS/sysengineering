"""Identity identifiers for secure platform (type-safe).

Uses Python's NewType for compile-time type safety without runtime overhead.

References:
    - design.md Section 2 (Architecture)
    - design.md Section 3 (Core Components)
"""

from __future__ import annotations

from typing import NewType

# SPIFFE URI format: spiffe://trust-domain/workload-type/workload-name
# Example: spiffe://cluster.local/service/payment-service

SpiffeId = NewType("SpiffeId", str)
CertificateSerial = NewType("CertificateSerial", str)
PolicyId = NewType("PolicyId", str)
PrincipalId = NewType("PrincipalId", str)
ResourceId = NewType("ResourceId", str)


def create_spiffe_id(workload_type: str, workload_name: str, trust_domain: str = "cluster.local") -> SpiffeId:
    """Create a SPIFFE ID.
    
    Args:
        workload_type: Type of workload (e.g., "service", "node", "pod").
        workload_name: Name of the workload (e.g., "payment-service").
        trust_domain: Trust domain (default: "cluster.local").
        
    Returns:
        SPIFFE ID in URI format.
    """
    return SpiffeId(f"spiffe://{trust_domain}/{workload_type}/{workload_name}")


def parse_spiffe_id(spiffe_id: SpiffeId) -> dict:
    """Parse a SPIFFE ID into components.
    
    Args:
        spiffe_id: SPIFFE ID to parse.
        
    Returns:
        Dictionary with trust_domain, workload_type, workload_name.
    """
    # Format: spiffe://trust-domain/workload-type/workload-name
    parts = spiffe_id.split('/')
    return {
        'trust_domain': parts[2],
        'workload_type': parts[3],
        'workload_name': parts[4],
    }


def create_certificate_serial() -> CertificateSerial:
    """Generate a unique certificate serial number.
    
    Returns:
        Certificate serial number.
    """
    import uuid
    return CertificateSerial(uuid.uuid4().hex)


def create_policy_id(name: str) -> PolicyId:
    """Create a policy ID.
    
    Args:
        name: Human-readable policy name.
        
    Returns:
        Policy ID.
    """
    return PolicyId(f"policy-{name}")


def create_principal_id(spiffe_id: SpiffeId) -> PrincipalId:
    """Create a principal ID from SPIFFE ID.
    
    Args:
        spiffe_id: SPIFFE ID of the principal.
        
    Returns:
        Principal ID.
    """
    return PrincipalId(str(spiffe_id))


def create_resource_id(resource_type: str, resource_name: str) -> ResourceId:
    """Create a resource ID.
    
    Args:
        resource_type: Type of resource (e.g., "database", "bucket", "api").
        resource_name: Name of the resource.
        
    Returns:
        Resource ID.
    """
    return ResourceId(f"{resource_type}/{resource_name}")
