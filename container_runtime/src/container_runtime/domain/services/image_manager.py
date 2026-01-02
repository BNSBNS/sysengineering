"""Image manager service."""

from __future__ import annotations

from typing import Optional
from container_runtime.domain.entities.container import ContainerImage, ImageFormat


class ImageManager:
    """Manages container images.
    
    Handles:
    - Image pulling/caching
    - Image listing
    - Image deletion
    - Image metadata
    """
    
    def __init__(self):
        """Initialize image manager."""
        self._images: dict[str, ContainerImage] = {}
    
    def pull(self, registry: str, name: str, tag: str = "latest") -> ContainerImage:
        """Pull (or get cached) image.
        
        Args:
            registry: Registry (e.g., "docker.io").
            name: Image name.
            tag: Image tag.
            
        Returns:
            Image.
        """
        image_key = f"{registry}/{name}:{tag}"
        
        # Check if already cached
        if image_key in self._images:
            return self._images[image_key]
        
        # Simulate pulling
        image = ContainerImage(
            image_id=image_key,
            name=name,
            tag=tag,
            registry=registry,
            format=ImageFormat.OCI,
            size_bytes=100_000_000,  # Simulate 100MB
            layer_count=5,
            digest=f"sha256:{'0' * 64}",
        )
        
        self._images[image_key] = image
        return image
    
    def get(self, image_id: str) -> Optional[ContainerImage]:
        """Get image by ID.
        
        Args:
            image_id: Image ID.
            
        Returns:
            Image or None.
        """
        return self._images.get(image_id)
    
    def list(self) -> list[ContainerImage]:
        """List all images.
        
        Returns:
            List of images.
        """
        return list(self._images.values())
    
    def delete(self, image_id: str) -> None:
        """Delete an image.
        
        Args:
            image_id: Image ID.
        """
        self._images.pop(image_id, None)
    
    def exists(self, image_id: str) -> bool:
        """Check if image exists.
        
        Args:
            image_id: Image ID.
            
        Returns:
            True if exists.
        """
        return image_id in self._images
    
    def get_size_mb(self, image_id: str) -> float:
        """Get image size in MB.
        
        Args:
            image_id: Image ID.
            
        Returns:
            Size in MB.
        """
        image = self._images.get(image_id)
        if not image:
            return 0
        return image.size_bytes / 1_000_000
