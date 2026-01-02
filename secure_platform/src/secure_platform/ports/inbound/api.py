"""Inbound ports (API contracts) for secure platform.

Defines the interface for external clients to interact with the secure platform.
Typically implemented by REST or gRPC adapters.

References:
    - design.md Section 2.4 (API Services)
    - Hexagonal Architecture pattern
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from secure_platform.domain.entities.certificate import Certificate
from secure_platform.domain.entities.policy import AuthorizationDecision, Principal


@dataclass
class IssueCertificateRequest:
    """Request to issue a certificate."""
    requester: str
    spiffe_id: str
    public_key_pem: str
    validity_days: int = 7


@dataclass
class IssueCertificateResponse:
    """Response with issued certificate."""
    certificate: Certificate
    success: bool
    message: str = ""


@dataclass
class AuthorizeRequest:
    """Request to authorize an action."""
    principal_id: str
    principal_type: str
    action: str
    resource_id: str
    roles: list[str] = None


@dataclass
class AuthorizeResponse:
    """Response with authorization decision."""
    decision: AuthorizationDecision
    success: bool
    message: str = ""


class CertificateAPI(ABC):
    """API for certificate operations."""
    
    @abstractmethod
    def issue_certificate(self, req: IssueCertificateRequest) -> IssueCertificateResponse:
        """Issue a new certificate.
        
        Args:
            req: Certificate request.
            
        Returns:
            Certificate response.
        """
        pass
    
    @abstractmethod
    def get_certificate(self, cert_serial: str) -> Certificate:
        """Get a certificate by serial.
        
        Args:
            cert_serial: Certificate serial number.
            
        Returns:
            Certificate.
        """
        pass


class AuthorizationAPI(ABC):
    """API for authorization decisions."""
    
    @abstractmethod
    def authorize(self, req: AuthorizeRequest) -> AuthorizeResponse:
        """Authorize an access request.
        
        Args:
            req: Authorization request.
            
        Returns:
            Authorization response.
        """
        pass


class AuditAPI(ABC):
    """API for audit log operations."""
    
    @abstractmethod
    def get_audit_stats(self) -> dict:
        """Get audit log statistics.
        
        Returns:
            Audit statistics.
        """
        pass
    
    @abstractmethod
    def get_root_hash(self) -> str:
        """Get audit log root hash (Merkle tree root).
        
        Used to verify log integrity.
        
        Returns:
            Root hash.
        """
        pass
