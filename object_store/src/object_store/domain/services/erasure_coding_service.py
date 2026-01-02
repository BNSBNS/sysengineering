"""Erasure coding service."""

from __future__ import annotations

from typing import Optional


class ErasureCodingService:
    """Erasure coding for fault tolerance."""
    
    def __init__(self, data_shards: int = 4, parity_shards: int = 2):
        """Initialize erasure coding.
        
        Args:
            data_shards: Number of data shards.
            parity_shards: Number of parity shards.
        """
        self.data_shards = data_shards
        self.parity_shards = parity_shards
        self.total_shards = data_shards + parity_shards
    
    def encode(self, data: bytes) -> list[bytes]:
        """Encode data with erasure coding.
        
        Args:
            data: Data to encode.
            
        Returns:
            List of shards.
        """
        # Simple implementation: split data across shards
        shard_size = (len(data) + self.data_shards - 1) // self.data_shards
        
        shards = []
        offset = 0
        
        # Data shards
        for i in range(self.data_shards):
            end = min(offset + shard_size, len(data))
            shard = data[offset:end]
            # Pad if needed
            if len(shard) < shard_size:
                shard = shard + b'\x00' * (shard_size - len(shard))
            shards.append(shard)
            offset = end
        
        # Parity shards (simple XOR for demo)
        for i in range(self.parity_shards):
            parity = bytearray(shard_size)
            for j in range(self.data_shards):
                for k in range(shard_size):
                    parity[k] ^= shards[j][k]
            shards.append(bytes(parity))
        
        return shards
    
    def decode(self, shards: list[bytes | None]) -> Optional[bytes]:
        """Decode data from shards.

        Args:
            shards: List of shards (None for missing shards).

        Returns:
            Decoded data or None if unrecoverable.
        """
        available = sum(1 for s in shards if s is not None)

        # Need at least data_shards total
        if available < self.data_shards:
            return None

        # Get shard size from first available shard
        shard_size = 0
        for s in shards:
            if s is not None:
                shard_size = len(s)
                break

        if shard_size == 0:
            return None

        # Check which data shards are missing
        missing_data_shards = []
        for i in range(self.data_shards):
            if shards[i] is None:
                missing_data_shards.append(i)

        # If we have all data shards, just concatenate
        if not missing_data_shards:
            data = b''.join(shards[i] for i in range(self.data_shards))
            return data.rstrip(b'\x00') if data else None

        # Try to recover missing data shards using parity
        # For XOR parity: parity = d0 ^ d1 ^ d2 ^ d3
        # So missing d2 = parity ^ d0 ^ d1 ^ d3
        recovered_shards = list(shards)

        for missing_idx in missing_data_shards:
            # Need a parity shard to recover
            parity_idx = None
            for i in range(self.data_shards, self.total_shards):
                if shards[i] is not None:
                    parity_idx = i
                    break

            if parity_idx is None:
                return None  # No parity shard available

            # XOR recovery: missing = parity ^ (all other data shards)
            recovered = bytearray(shards[parity_idx])
            for i in range(self.data_shards):
                if i != missing_idx and recovered_shards[i] is not None:
                    for k in range(shard_size):
                        recovered[k] ^= recovered_shards[i][k]

            recovered_shards[missing_idx] = bytes(recovered)

        # Concatenate recovered data shards
        data = b''.join(recovered_shards[i] for i in range(self.data_shards))
        return data.rstrip(b'\x00') if data else None
    
    def can_recover(self, available_shards: int) -> bool:
        """Check if data can be recovered.
        
        Args:
            available_shards: Number of available shards.
            
        Returns:
            True if recoverable.
        """
        return available_shards >= self.data_shards
    
    def min_shards_required(self) -> int:
        """Minimum shards needed for recovery.
        
        Returns:
            Minimum shard count.
        """
        return self.data_shards
