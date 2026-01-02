"""X.509 certificate entities for PKI.

Represents certificates used for mTLS, with lifecycle management
(issue, validate, rotate, revoke).

References:
    - design.md Section 3 (PKI Manager)
    - design.md Section 4 (Certificate Lifecycle)
    - RFC 5280 (X.509 PKI)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from secure_platform.domain.value_objects.identifiers import (
    CertificateSerial,
    SpiffeId,
    create_certificate_serial,
)


class CertificateState(Enum):
    """Certificate lifecycle states."""
    PENDING = "pending"        # Issued but not yet active
    VALID = "valid"            # Active and usable
    ROTATING = "rotating"      # Being rotated to new cert
    EXPIRED = "expired"        # Valid period ended
    REVOKED = "revoked"        # Explicitly revoked
    COMPROMISED = "compromised"  # Key material compromised


@dataclass
class CertificateSpecs:
    """X.509 certificate specification."""
    serial: CertificateSerial
    spiffe_id: SpiffeId
    public_key_pem: str  # PEM-encoded RSA/EC public key
    valid_from: datetime
    valid_until: datetime
    issuer_cn: str = "SecurePlatform-CA"  # Certificate Authority name
    signature_algorithm: str = "sha256WithRSAEncryption"


@dataclass
class Certificate:
    """X.509 certificate entity."""
    specs: CertificateSpecs
    state: CertificateState = CertificateState.VALID
    created_timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())
    revoked_timestamp: Optional[float] = None
    revocation_reason: Optional[str] = None
    
    @property
    def is_valid(self) -> bool:
        """Check if certificate is currently valid."""
        now = datetime.utcnow()
        return (
            self.state == CertificateState.VALID
            and self.specs.valid_from <= now <= self.specs.valid_until
        )
    
    @property
    def is_expired(self) -> bool:
        """Check if certificate has expired."""
        return datetime.utcnow() > self.specs.valid_until
    
    @property
    def days_until_expiry(self) -> float:
        """Days until certificate expires.
        
        Returns:
            Positive if valid, negative if expired.
        """
        now = datetime.utcnow()
        delta = self.specs.valid_until - now
        return delta.total_seconds() / 86400
    
    def revoke(self, reason: str) -> None:
        """Revoke the certificate.
        
        Args:
            reason: Revocation reason (e.g., "key compromised").
        """
        self.state = CertificateState.REVOKED
        self.revoked_timestamp = datetime.utcnow().timestamp()
        self.revocation_reason = reason
    
    def mark_rotating(self) -> None:
        """Mark certificate as being rotated to new cert."""
        self.state = CertificateState.ROTATING
    
    def mark_expired(self) -> None:
        """Mark certificate as expired."""
        self.state = CertificateState.EXPIRED


@dataclass
class CertificateSigningRequest:
    """CSR (Certificate Signing Request) for requesting a certificate."""
    spiffe_id: SpiffeId
    public_key_pem: str  # PEM-encoded RSA/EC public key
    validity_days: int = 7  # NIST recommends short-lived certs (hours to days)
    
    def create_certificate(self) -> Certificate:
        """Create a certificate from this CSR.
        
        Returns:
            Certificate with VALID state.
        """
        now = datetime.utcnow()
        specs = CertificateSpecs(
            serial=create_certificate_serial(),
            spiffe_id=self.spiffe_id,
            public_key_pem=self.public_key_pem,
            valid_from=now,
            valid_until=now + timedelta(days=self.validity_days),
        )
        return Certificate(specs=specs, state=CertificateState.VALID)


@dataclass
class CertificateChain:
    """Chain of certificates for mTLS handshake.
    
    Typically: [Leaf Cert] + [Intermediate CA] + [Root CA]
    """
    leaf_cert: Certificate
    intermediate_certs: list[Certificate] = field(default_factory=list)
    root_cert: Optional[Certificate] = None
    
    def verify_chain_validity(self) -> bool:
        """Verify entire certificate chain is valid.
        
        Returns:
            True if all certificates in chain are valid.
        """
        return (
            self.leaf_cert.is_valid
            and all(cert.is_valid for cert in self.intermediate_certs)
            and (self.root_cert is None or self.root_cert.is_valid)
        )
