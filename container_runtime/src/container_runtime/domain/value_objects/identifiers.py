"""Container runtime value objects."""

from typing import NewType

# Type-safe identifiers
ContainerId = NewType('ContainerId', str)
ImageId = NewType('ImageId', str)
JobId = NewType('JobId', str)
NamespaceId = NewType('NamespaceId', str)
CgroupId = NewType('CgroupId', str)
GPUId = NewType('GPUId', str)


def create_container_id(image_name: str, instance: int) -> ContainerId:
    """Create a container ID.
    
    Args:
        image_name: Container image name.
        instance: Instance number.
        
    Returns:
        Container ID.
    """
    return ContainerId(f"{image_name}-{instance}")


def create_job_id(user: str, timestamp: int) -> JobId:
    """Create a job ID.
    
    Args:
        user: User submitting job.
        timestamp: Job submission timestamp.
        
    Returns:
        Job ID.
    """
    return JobId(f"job-{user}-{timestamp}")


def create_image_id(name: str, tag: str) -> ImageId:
    """Create an image ID.
    
    Args:
        name: Image name.
        tag: Image tag.
        
    Returns:
        Image ID.
    """
    return ImageId(f"{name}:{tag}")


def create_gpu_id(index: int) -> GPUId:
    """Create a GPU ID.
    
    Args:
        index: GPU index from NVML.
        
    Returns:
        GPU ID.
    """
    return GPUId(f"gpu-{index}")
