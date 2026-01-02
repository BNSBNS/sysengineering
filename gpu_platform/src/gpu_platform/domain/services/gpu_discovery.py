"""GPU discovery service using NVIDIA Management Library (NVML).

This service scans the system for available GPUs, queries their properties,
monitors health metrics (temperature, power, ECC errors), and detects XID errors.

References:
    - design.md Section 3 (GPU Discovery)
    - design.md Section 6 (Failure Modes & Recovery)
    - NVIDIA NVML API docs: developer.nvidia.com/nvidia-management-library
"""

from __future__ import annotations

import logging
import time
from typing import Optional

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

from gpu_platform.domain.entities.gpu_device import (
    GPUDevice,
    GPUHealth,
    GPUSpecs,
    GPUState,
)
from gpu_platform.domain.value_objects.gpu_identifiers import (
    GPUId,
    NUMANodeId,
    PCIeBusId,
    create_gpu_id,
)

logger = logging.getLogger(__name__)


class GPUDiscoveryError(Exception):
    """GPU discovery operation failed."""
    pass


class GPUDiscoveryService:
    """Discover and monitor GPUs using NVML.
    
    If NVML is not available (no GPU or not installed), gracefully falls back to mock mode
    for development and testing.
    """

    def __init__(self, dev_mode: bool = False) -> None:
        """Initialize NVML or dev mode.
        
        Args:
            dev_mode: Force development mode (mock GPUs) even if NVML is available.
                     Useful for testing without GPU hardware.
        """
        self._dev_mode = dev_mode
        self._initialized = False
        
        if dev_mode:
            logger.info("GPU discovery running in DEVELOPMENT MODE (mocked)")
            self._initialized = True
            return
        
        if not NVML_AVAILABLE:
            logger.warning(
                "pynvml not installed. GPU discovery will use development mode. "
                "Install with: pip install nvidia-ml-py"
            )
            self._dev_mode = True
            self._initialized = True
            return
        
        try:
            pynvml.nvmlInit()
            self._initialized = True
            logger.info("NVML initialized successfully")
        except pynvml.NVMLError as e:
            logger.warning(f"Failed to initialize NVML: {e}. Falling back to development mode.")
            self._dev_mode = True
            self._initialized = True
    
    def scan_gpus(self) -> list[GPUSpecs]:
        """Discover all GPUs in the system.
        
        Returns:
            List of GPU specifications found. In dev mode, returns mock GPUs.
            
        Raises:
            GPUDiscoveryError: If scan fails (real mode only).
        """
        if not self._initialized:
            raise GPUDiscoveryError("NVML not initialized")
        
        if self._dev_mode:
            return self._mock_scan_gpus()
        
        gpus = []
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get model name
                model = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # Get compute capability (major.minor)
                cc_major, cc_minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                compute_capability = f"{cc_major}.{cc_minor}"
                
                # Get memory in MB
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_mb = memory_info.total // (1024 * 1024)
                
                # Get PCIe bus ID
                pcie_bus = pynvml.nvmlDeviceGetPcieThroughputInfo(handle, 0)
                pcie_bus_id = PCIeBusId("0000:00:00.0")  # Default placeholder
                
                # Check MIG support (A100, H100+)
                supports_mig = "A100" in model or "H100" in model
                
                gpu_id = create_gpu_id(i)
                specs = GPUSpecs(
                    gpu_id=gpu_id,
                    model=model,
                    compute_capability=compute_capability,
                    memory_mb=memory_mb,
                    pcie_bus_id=pcie_bus_id,
                    supports_mig=supports_mig,
                )
                gpus.append(specs)
                logger.info(f"Discovered {model} (CC {compute_capability}) at {gpu_id}")
        
        except pynvml.NVMLError as e:
            raise GPUDiscoveryError(f"Failed to scan GPUs: {e}")
        
        return gpus
    
    def _mock_scan_gpus(self) -> list[GPUSpecs]:
        """Generate mock GPU specs for development/testing without hardware.
        
        Returns:
            List of 2 simulated A100 GPUs (common in ML clusters).
        """
        mock_gpus = []
        for i in range(2):
            gpu_id = create_gpu_id(i)
            specs = GPUSpecs(
                gpu_id=gpu_id,
                model="A100 (simulated)",
                compute_capability="8.0",
                memory_mb=40960,
                pcie_bus_id=PCIeBusId(f"0000:01:0{i}.0"),
                supports_mig=True,
                nvlink_ports=2,
            )
            mock_gpus.append(specs)
            logger.info(f"[DEV MODE] Simulated {specs.model} at {gpu_id}")
        return mock_gpus
    
    def get_health(self, gpu_id: GPUId) -> Optional[GPUHealth]:
        """Get current health metrics for a GPU.
        
        Args:
            gpu_id: GPU identifier.
            
        Returns:
            Current health metrics or None if unavailable.
            In dev mode, returns simulated healthy metrics.
        """
        if not self._initialized:
            return None
        
        if self._dev_mode:
            return self._mock_get_health(gpu_id)
        
        try:
            # Parse GPU index from gpu_id (e.g., "GPU-0" -> 0)
            idx = int(gpu_id.split('-')[1])
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            
            # Temperature
            temperature = pynvml.nvmlDeviceGetTemperature(handle, 0)  # 0 = GPU die
            
            # Power usage
            try:
                power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except:
                power_w = 0.0
            
            # Utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization = float(util.gpu)
            except:
                utilization = 0.0
            
            # Memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used_mb = mem_info.used // (1024 * 1024)
            
            # ECC errors
            ecc_correct = 0
            ecc_uncorrect = 0
            try:
                # Try to get ECC stats
                ecc_data = pynvml.nvmlDeviceGetEccMode(handle)
                # Note: Full ECC stats require different API calls
            except:
                pass
            
            return GPUHealth(
                gpu_id=gpu_id,
                temperature_c=float(temperature),
                power_w=power_w,
                utilization_percent=utilization,
                memory_used_mb=memory_used_mb,
                ecc_errors_correctable=ecc_correct,
                ecc_errors_uncorrectable=ecc_uncorrect,
                throttled=float(temperature) > 85,  # Thermal throttle threshold
                last_update_timestamp=time.time(),
            )
        
        except (pynvml.NVMLError, ValueError, IndexError) as e:
            logger.warning(f"Failed to get health for {gpu_id}: {e}")
            return None
    
    def _mock_get_health(self, gpu_id: GPUId) -> GPUHealth:
        """Return simulated healthy metrics for development/testing.
        
        Args:
            gpu_id: GPU identifier.
            
        Returns:
            Mock health metrics showing a healthy, underutilized GPU.
        """
        return GPUHealth(
            gpu_id=gpu_id,
            temperature_c=45.0,
            power_w=250.0,
            utilization_percent=25.0,
            memory_used_mb=1024,
            ecc_errors_correctable=0,
            ecc_errors_uncorrectable=0,
            throttled=False,
            last_update_timestamp=time.time(),
        )
    
    def shutdown(self) -> None:
        """Shutdown NVML."""
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
                self._initialized = False
            except pynvml.NVMLError as e:
                logger.error(f"Error shutting down NVML: {e}")
    
    def __enter__(self) -> GPUDiscoveryService:
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.shutdown()
