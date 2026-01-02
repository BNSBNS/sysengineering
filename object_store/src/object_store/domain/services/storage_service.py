"""Storage service for object management."""

from __future__ import annotations

import hashlib
from typing import Optional

from object_store.domain.entities.bucket import Bucket
from object_store.domain.entities.chunk import Chunk, ChunkRef
from object_store.domain.entities.object import Object, ObjectVersion


class StorageService:
    """Core storage logic for objects and chunks."""
    
    def __init__(self, chunk_size: int = 64 * 1024 * 1024):
        """Initialize storage service.
        
        Args:
            chunk_size: Default chunk size in bytes.
        """
        self.chunk_size = chunk_size
        self.chunks = {}  # chunk_id -> Chunk
        self.objects = {}  # object_id -> Object
    
    def put_object(
        self,
        bucket: Bucket,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> Object:
        """Put an object in storage.
        
        Args:
            bucket: Target bucket.
            key: Object key.
            data: Object data.
            content_type: MIME type.
            
        Returns:
            Stored object.
        """
        # Create object
        import uuid
        object_id = f"obj-{uuid.uuid4().hex[:12]}"
        obj = Object(
            object_id=object_id,
            bucket=bucket.name,
            key=key,
            size=len(data),
            content_type=content_type,
            checksum=hashlib.sha256(data).hexdigest(),
        )
        
        # Split into chunks
        chunk_refs = self._chunk_data(obj, data, bucket.get_chunk_size())
        obj.chunk_refs = chunk_refs
        
        # Store
        self.objects[object_id] = obj
        return obj
    
    def get_object(self, object_id: str) -> Optional[Object]:
        """Get object metadata.
        
        Args:
            object_id: Object ID.
            
        Returns:
            Object or None if not found.
        """
        return self.objects.get(object_id)
    
    def delete_object(self, object_id: str) -> bool:
        """Delete an object.
        
        Args:
            object_id: Object ID.
            
        Returns:
            True if deleted.
        """
        if object_id in self.objects:
            del self.objects[object_id]
            return True
        return False
    
    def _chunk_data(self, obj: Object, data: bytes, chunk_size: int) -> list[ChunkRef]:
        """Split data into chunks.
        
        Args:
            obj: Parent object.
            data: Data to chunk.
            chunk_size: Size per chunk.
            
        Returns:
            List of chunk references.
        """
        import uuid
        
        chunk_refs = []
        offset = 0
        sequence = 0
        
        while offset < len(data):
            end = min(offset + chunk_size, len(data))
            chunk_data = data[offset:end]
            
            chunk_id = f"chunk-{uuid.uuid4().hex[:12]}"
            checksum = hashlib.sha256(chunk_data).hexdigest()
            
            chunk = Chunk(
                chunk_id=chunk_id,
                object_id=obj.object_id,
                sequence=sequence,
                data=chunk_data,
                size=len(chunk_data),
                checksum=checksum,
            )
            
            self.chunks[chunk_id] = chunk
            
            ref = ChunkRef(
                chunk_id=chunk_id,
                checksum=checksum,
                sequence=sequence,
                size=len(chunk_data),
            )
            chunk_refs.append(ref)
            
            offset = end
            sequence += 1
        
        return chunk_refs
    
    def verify_object_integrity(self, object_id: str) -> bool:
        """Verify object integrity via chunk checksums.
        
        Args:
            object_id: Object ID.
            
        Returns:
            True if all chunks valid.
        """
        obj = self.objects.get(object_id)
        if not obj:
            return False
        
        for ref in obj.chunk_refs:
            chunk = self.chunks.get(ref.chunk_id)
            if not chunk or not chunk.verify_checksum():
                return False
        
        return True
