"""Domain services."""

from object_store.domain.services.storage_service import StorageService
from object_store.domain.services.erasure_coding_service import ErasureCodingService

__all__ = [
    "StorageService",
    "ErasureCodingService",
]
