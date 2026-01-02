"""PKI (Public Key Infrastructure) management service.

Issues, validates, rotates, and revokes X.509 certificates for mTLS.

References:
    - design.md Section 3 (PKI Manager)
    - RFC 5280 (X.509 PKI)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from secure_platform.domain.entities.certificate import (
    Certificate,
    CertificateSigningRequest,
    CertificateState,
)
from secure_platform.domain.value_objects.identifiers import (
    CertificateSerial,
    SpiffeId,
)

logger = logging.getLogger(__name__)


class PKIError(Exception):
    """PKI operation failed."""
    pass


class PKIManager:
    """Manage X.509 certificates for mTLS."""
    
    def __init__(self):
        """Initialize PKI manager."""
        self._certificates: dict[CertificateSerial, Certificate] = {}
        self._revoked_serials: set[CertificateSerial] = set()
        self._spiffe_to_cert: dict[SpiffeId, CertificateSerial] = {}
    
    def issue_certificate(self, csr: CertificateSigningRequest) -> Certificate:
        """Issue a new certificate from CSR.
        
        Args:
            csr: Certificate signing request.
            
        Returns:
            Issued certificate.
            
        Raises:
            PKIError: If CSR is invalid.
        """
        if not csr.public_key_pem or len(csr.public_key_pem) < 10:
            raise PKIError("Invalid public key in CSR")
        
        try:
            cert = csr.create_certificate()
            self._certificates[cert.specs.serial] = cert
            self._spiffe_to_cert[csr.spiffe_id] = cert.specs.serial
            
            logger.info(f"Issued certificate for {csr.spiffe_id} (serial={cert.specs.serial})")
            return cert
        
        except Exception as e:
            raise PKIError(f"Failed to issue certificate: {e}")
    
    def validate_certificate(self, cert: Certificate) -> bool:
        """Validate a certificate.
        
        Args:
            cert: Certificate to validate.
            
        Returns:
            True if certificate is valid.
        """
        # Check state
        if cert.state != CertificateState.VALID:
            return False
        
        # Check revocation
        if cert.specs.serial in self._revoked_serials:
            return False
        
        # Check expiry
        if cert.is_expired:
            cert.mark_expired()
            return False
        
        return True
    
    def revoke_certificate(self, serial: CertificateSerial, reason: str = "") -> None:
        """Revoke a certificate.
        
        Args:
            serial: Serial number of certificate to revoke.
            reason: Revocation reason.
        """
        if serial not in self._certificates:
            raise PKIError(f"Certificate {serial} not found")
        
        cert = self._certificates[serial]
        cert.revoke(reason)
        self._revoked_serials.add(serial)
        
        logger.warning(f"Revoked certificate {serial} (reason: {reason})")
    
    def rotate_certificate(self, old_cert: Certificate) -> Certificate:
        """Rotate a certificate to a new one.
        
        Creates a new certificate with the same SPIFFE ID.
        
        Args:
            old_cert: Certificate to rotate.
            
        Returns:
            New certificate.
        """
        if not self.validate_certificate(old_cert):
            raise PKIError(f"Cannot rotate invalid certificate {old_cert.specs.serial}")
        
        # Create CSR with same SPIFFE ID and key
        csr = CertificateSigningRequest(
            spiffe_id=old_cert.specs.spiffe_id,
            public_key_pem=old_cert.specs.public_key_pem,
            validity_days=7,
        )
        
        # Issue new certificate
        new_cert = self.issue_certificate(csr)
        
        # Mark old as rotating
        old_cert.mark_rotating()
        
        logger.info(
            f"Rotated certificate for {old_cert.specs.spiffe_id} "
            f"(old={old_cert.specs.serial}, new={new_cert.specs.serial})"
        )
        
        return new_cert
    
    def get_certificate(self, serial: CertificateSerial) -> Optional[Certificate]:
        """Get a certificate by serial number.
        
        Args:
            serial: Certificate serial number.
            
        Returns:
            Certificate or None if not found.
        """
        return self._certificates.get(serial)
    
    def get_current_certificate(self, spiffe_id: SpiffeId) -> Optional[Certificate]:
        """Get current valid certificate for SPIFFE ID.
        
        Args:
            spiffe_id: SPIFFE ID of the principal.
            
        Returns:
            Current certificate or None if not found.
        """
        serial = self._spiffe_to_cert.get(spiffe_id)
        if not serial:
            return None
        
        cert = self._certificates.get(serial)
        if cert and self.validate_certificate(cert):
            return cert
        
        return None
    
    def get_stats(self) -> dict:
        """Get PKI statistics.
        
        Returns:
            Dictionary with certificate stats.
        """
        valid_count = sum(1 for c in self._certificates.values() if c.state == CertificateState.VALID)
        revoked_count = len(self._revoked_serials)
        expired_count = sum(1 for c in self._certificates.values() if c.state == CertificateState.EXPIRED)
        
        return {
            'total_issued': len(self._certificates),
            'valid': valid_count,
            'revoked': revoked_count,
            'expired': expired_count,
            'unique_principals': len(self._spiffe_to_cert),
        }
