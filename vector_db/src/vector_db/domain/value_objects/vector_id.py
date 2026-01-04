"""Vector ID value object for type-safe vector identification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class VectorId:
    """Immutable, type-safe vector identifier.

    Provides:
    - Type safety: Cannot accidentally use raw strings/ints as IDs
    - Immutability: IDs cannot be modified after creation
    - Hashability: Can be used in sets and as dict keys
    """

    value: str

    def __post_init__(self) -> None:
        """Validate the ID value."""
        if not self.value:
            raise ValueError("VectorId cannot be empty")
        if len(self.value) > 256:
            raise ValueError("VectorId cannot exceed 256 characters")

    @classmethod
    def generate(cls) -> Self:
        """Generate a new random VectorId using UUID4."""
        return cls(str(uuid4()))

    @classmethod
    def from_uuid(cls, uuid: UUID) -> Self:
        """Create a VectorId from a UUID."""
        return cls(str(uuid))

    @classmethod
    def from_int(cls, value: int) -> Self:
        """Create a VectorId from an integer (for sequential IDs)."""
        return cls(str(value))

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"VectorId({self.value!r})"
